"""Тесты для GoogleGenAIProvider, разбора ошибок, dynamic timeout и thinking fallback."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from chllm.exceptions import (
    AuthenticationError,
    CHLLMError,
    ContentBlockedError,
    LocationUnsupportedError,
    ProxyConnectionError,
    QuotaExceededError,
    RateLimitError,
    ServiceUnavailableError,
)
from chllm.providers import (
    GeminiThinkingConfig,
    GoogleGenAIProvider,
    model_supports_thinking,
)


def test_model_supports_thinking():
    """Проверка определения поддержки thinking у различных моделей."""
    assert model_supports_thinking("gemini-2.5-flash") is True
    assert model_supports_thinking("gemini-2.5-pro") is True
    assert model_supports_thinking("gemini-3.0-flash") is True
    assert model_supports_thinking("gemini-3-pro") is True
    assert model_supports_thinking("custom-thinking-model") is True

    assert model_supports_thinking("gemini-1.5-flash") is False
    assert model_supports_thinking("gemini-1.5-pro") is False
    assert model_supports_thinking("gpt-4o") is False
    assert model_supports_thinking("") is False


def test_calculate_dynamic_timeout():
    """Проверка расчета динамического таймаута с ограничениями min/max."""
    provider = GoogleGenAIProvider(
        client=MagicMock(),
        base_timeout=120.0,
        min_timeout=180.0,
        max_timeout=1200.0,
        tokens_per_second=18.0,
    )

    # 1000 токенов: 120 + 1000/18 = ~175.5 -> ограничен min_timeout (180.0)
    assert provider.calculate_dynamic_timeout(1000) == 180.0

    # 5000 токенов: 120 + 5000/18 = ~397.7
    t5k = provider.calculate_dynamic_timeout(5000)
    assert 395.0 < t5k < 400.0

    # 50000 токенов: 120 + 50000/18 = ~2897 -> ограничен max_timeout (1200.0)
    assert provider.calculate_dynamic_timeout(50000) == 1200.0


def test_handle_api_error_recursive_unwrapping_network():
    """Проверка рекурсивного разбора вложенных сетевых исключений."""
    provider = GoogleGenAIProvider(client=MagicMock())

    class CustomRemoteProtocolError(Exception):
        pass

    # Ошибка завернута в цепочку причин
    root_cause = CustomRemoteProtocolError("Connection reset by peer")
    wrapper_err = RuntimeError("SDK Error")
    wrapper_err.__cause__ = root_cause

    with pytest.raises(ProxyConnectionError) as exc_info:
        provider.handle_api_error(wrapper_err)

    assert "CustomRemoteProtocolError" in str(exc_info.value) or "сетевая ошибка" in str(exc_info.value)


def test_handle_api_error_timeout_and_cancellation():
    """Проверка обработки TimeoutError и CancelledError."""
    provider = GoogleGenAIProvider(client=MagicMock())

    with pytest.raises(ProxyConnectionError):
        provider.handle_api_error(TimeoutError("Request timed out"))

    with pytest.raises(ProxyConnectionError):
        provider.handle_api_error(asyncio.CancelledError("Operation cancelled"))


def test_handle_api_error_authentication():
    """Проверка детекции неверного API-ключа."""
    provider = GoogleGenAIProvider(client=MagicMock())

    err = Exception("API_KEY_INVALID: Provided API key is not valid")
    with pytest.raises(AuthenticationError):
        provider.handle_api_error(err)


def test_handle_api_error_geolocation():
    """Проверка детекции геоблокировок."""
    provider = GoogleGenAIProvider(client=MagicMock())

    err = Exception("User location is not supported for the requested model.")
    with pytest.raises(LocationUnsupportedError):
        provider.handle_api_error(err)


def test_handle_api_error_daily_quota():
    """Проверка детекции исчерпания суточной квоты."""
    provider = GoogleGenAIProvider(client=MagicMock())

    err = Exception("Resource has been exhausted (e.g. check quota). Daily quota exceeded per day.")
    with pytest.raises(QuotaExceededError):
        provider.handle_api_error(err)


def test_handle_api_error_rate_limit_with_retry_delay():
    """Проверка извлечения серверного retryDelay при 429 RPM/TPM."""
    provider = GoogleGenAIProvider(client=MagicMock())

    # Сервер возвращает 429 с retryDelay: 120s
    err = Exception("429 Resource exhausted: Rate limit reached. 'retryDelay': '120s'")
    with pytest.raises(RateLimitError) as exc_info:
        provider.handle_api_error(err)

    assert exc_info.value.retry_delay == 120.0

    # Сервер возвращает 429 с retryDelay: 45
    err2 = Exception('{"error": {"code": 429, "message": "Too Many Requests", "retryDelay": "45s"}}')
    with pytest.raises(RateLimitError) as exc_info2:
        provider.handle_api_error(err2)

    assert exc_info2.value.retry_delay == 45.0


def test_handle_api_error_service_unavailable():
    """Проверка ошибки 503 с retryDelay."""
    provider = GoogleGenAIProvider(client=MagicMock())

    err = Exception("503 The model is overloaded. Please try again later. retryDelay: 15s")
    with pytest.raises(ServiceUnavailableError) as exc_info:
        provider.handle_api_error(err)

    assert exc_info.value.retry_delay == 15.0


def test_handle_api_error_content_blocked():
    """Проверка детекции блокировки контента."""
    provider = GoogleGenAIProvider(client=MagicMock())

    err = Exception("Candidate was blocked due to safety policies: prompt_feedback")
    with pytest.raises(ContentBlockedError):
        provider.handle_api_error(err)


def test_handle_api_error_generic_fallback():
    """Проверка проброса общего исключения CHLLMError."""
    provider = GoogleGenAIProvider(client=MagicMock())

    err = ValueError("Something unexpected happened")
    with pytest.raises(CHLLMError):
        provider.handle_api_error(err)


@pytest.mark.asyncio
async def test_generate_and_execute_success():
    """Проверка успешной генерации контента и вызова через протокол execute."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Ответ от Gemini"

    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    provider = GoogleGenAIProvider(
        client=mock_client,
        model="gemini-2.5-flash",
        system_instruction="Ты помощник.",
    )

    result = await provider.generate("Привет")
    assert result == "Ответ от Gemini"

    # Проверка через execute с dict
    dict_result = await provider.execute({"prompt": "Тест в словаре"})
    assert dict_result == "Ответ от Gemini"

    # Проверка через execute со списком
    list_result = await provider.execute(["Часть 1", "Часть 2"])
    assert list_result == "Ответ от Gemini"

    assert provider.pack_single_prompt("Промпт") == "Промпт"


@pytest.mark.asyncio
async def test_generate_thinking_fallback():
    """Проверка автоматического повтора без thinking при ошибке модели."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Ответ после fallback"

    # Первый вызов падает с ошибкой о thinking_config, второй вызов успешен
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=[
            ValueError("thinking_config is not supported for this model"),
            mock_response,
        ]
    )

    thinking_cfg = GeminiThinkingConfig(thinking_level="HIGH", thinking_budget=1024)
    provider = GoogleGenAIProvider(
        client=mock_client,
        model="gemini-2.5-flash",
        thinking_config=thinking_cfg,
    )

    result = await provider.generate("Сложный запрос")
    assert result == "Ответ после fallback"
    assert mock_client.aio.models.generate_content.call_count == 2


@pytest.mark.asyncio
async def test_generate_empty_response_blocks():
    """Проверка генерации исключения при пустом ответе модели."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = ""
    mock_response.prompt_feedback = "HARM_CATEGORY_DANGEROUS_CONTENT"

    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    provider = GoogleGenAIProvider(client=mock_client)

    with pytest.raises(ContentBlockedError) as exc_info:
        await provider.generate("Опасный запрос")

    assert "HARM_CATEGORY_DANGEROUS_CONTENT" in str(exc_info.value)


@pytest.mark.asyncio
async def test_check_connection_and_close():
    """Проверка методов check_connection и close."""
    mock_client = MagicMock()

    async def fake_model_iter():
        yield MagicMock(name="gemini-2.5-flash")

    mock_client.aio.models.list = AsyncMock(return_value=fake_model_iter())
    mock_client.aio.aclose = AsyncMock()

    provider = GoogleGenAIProvider(client=mock_client)

    ok, msg = await provider.check_connection()
    assert ok is True
    assert "установлено" in msg

    await provider.close()
    mock_client.aio.aclose.assert_called_once()
