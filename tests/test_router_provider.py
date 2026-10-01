"""Тесты для OpenAIRouterProvider, работы со шлюзами, сессиями BLG и метриками."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from chllm import (
    AuthenticationError,
    CHLLMError,
    ContentBlockedError,
    LocationUnsupportedError,
    OpenAIRouterProvider,
    ProxyConnectionError,
    QuotaExceededError,
    RateLimitError,
    ServiceUnavailableError,
    mask_proxy_url,
)


def test_mask_proxy_url():
    """Проверка корректной маскировки учетных данных в URL прокси."""
    assert mask_proxy_url("") == ""
    assert mask_proxy_url(None) == ""
    assert mask_proxy_url("http://user:secret123@127.0.0.1:8080") == "http://user:***@127.0.0.1:8080"
    assert mask_proxy_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"


def test_request_headers_generation():
    """Проверка формирования HTTP-заголовков для стандартного роутера и BLG."""
    # Стандартный роутер
    provider = OpenAIRouterProvider(
        base_url="https://api.openai.com/v1",
        api_key="sk-real-key",
        is_blg=False,
        session_id="custom-session-123",
        extra_headers={"X-Custom-App": "test-app"},
    )
    headers = provider._get_request_headers()
    assert headers["Authorization"] == "Bearer sk-real-key"
    assert headers["X-Session-ID"] == "custom-session-123"
    assert headers["X-Custom-App"] == "test-app"
    assert "X-Input-Mode" not in headers

    # BLG шлюз (Browser LLM Gateway)
    blg_provider = OpenAIRouterProvider(
        base_url="http://127.0.0.1:8000/v1",
        is_blg=True,
        session_id="blg-session-456",
    )
    blg_headers = blg_provider._get_request_headers("blg-session-456")
    assert blg_headers["X-Input-Mode"] == "paste"
    assert blg_headers["X-Paste-Threshold"] == "0"
    assert blg_headers["X-Temporary-Chat"] == "true"
    assert blg_headers["X-Session-ID"] == "blg-session-456"
    assert blg_headers["Authorization"] == "Bearer blg-session-456"


@pytest.mark.asyncio
async def test_generate_success_and_usage_metrics():
    """Проверка успешной генерации и извлечения UsageMetrics."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Ответ от шлюза",
                }
            }
        ],
        "usage": {
            "prompt_tokens": 42,
            "completion_tokens": 18,
            "total_tokens": 60,
        },
    }
    mock_client.post = AsyncMock(return_value=mock_response)

    provider = OpenAIRouterProvider(
        base_url="http://127.0.0.1:8000/v1",
        client=mock_client,
        is_blg=False,
    )

    result = await provider.generate("Привет, шлюз!")
    assert result == "Ответ от шлюза"

    # Проверка метрик
    assert provider.last_usage is not None
    assert provider.last_usage.prompt_tokens == 42
    assert provider.last_usage.completion_tokens == 18
    assert provider.last_usage.total_tokens == 60

    # Проверка вызова POST /v1/chat/completions
    mock_client.post.assert_called_once()
    called_url = mock_client.post.call_args[0][0]
    assert called_url == "http://127.0.0.1:8000/v1/chat/completions"


@pytest.mark.asyncio
async def test_execute_and_pack_prompt():
    """Проверка протокола execute с различными форматами входных данных."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "Результат"}}]}
    mock_client.post = AsyncMock(return_value=mock_response)

    provider = OpenAIRouterProvider(client=mock_client, is_blg=False)

    # Строка
    res_str = await provider.execute("Промпт")
    assert res_str == "Результат"

    # Список
    res_list = await provider.execute(["Строка 1", "Строка 2"])
    assert res_list == "Результат"

    # Словарь
    res_dict = await provider.execute({"prompt": "В словаре", "system_prompt": "Инструкция"})
    assert res_dict == "Результат"

    assert provider.pack_single_prompt("Тест") == "Тест"


@pytest.mark.asyncio
async def test_blg_session_cleanup():
    """Проверка вызова DELETE /sessions/{id} для BLG шлюза."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "BLG ответ"}}]}
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.delete = AsyncMock(return_value=MagicMock(status_code=200))
    mock_client.is_closed = False

    provider = OpenAIRouterProvider(
        base_url="http://127.0.0.1:8000/v1",
        client=mock_client,
        is_blg=True,
    )

    result = await provider.generate("Запрос в BLG")
    assert result == "BLG ответ"

    # Проверяем, что DELETE был вызван в блоке finally
    mock_client.delete.assert_called_once()
    delete_url = mock_client.delete.call_args[0][0]
    assert "/v1/sessions/chllm-" in delete_url or "/sessions/chllm-" in delete_url


def test_handle_api_error_network():
    """Проверка преобразования сетевых ошибок httpx."""
    provider = OpenAIRouterProvider(client=MagicMock())

    req = httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions")
    err = httpx.ConnectError("Failed to establish a connection", request=req)

    with pytest.raises(ProxyConnectionError):
        provider.handle_api_error(err)

    with pytest.raises(ProxyConnectionError):
        provider.handle_api_error(asyncio.TimeoutError("Timeout"))


def test_handle_api_error_http_status_codes():
    """Проверка маппинга HTTP-статусов (401, 403, 429, 500, 502, 503, 504)."""
    provider = OpenAIRouterProvider(client=MagicMock())
    req = httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions")

    # 401 Unauthorized
    resp_401 = httpx.Response(401, text="Unauthorized: Invalid key", request=req)
    with pytest.raises(AuthenticationError):
        provider.handle_api_error(httpx.HTTPStatusError("Auth error", request=req, response=resp_401))

    # 429 Rate Limit с retryDelay
    resp_429 = httpx.Response(429, text='{"error": "Too Many Requests", "retryDelay": "45s"}', request=req)
    with pytest.raises(RateLimitError) as exc_429:
        provider.handle_api_error(httpx.HTTPStatusError("Rate limit", request=req, response=resp_429))
    assert exc_429.value.retry_delay == 45.0

    # 429 Суточная квота
    resp_quota = httpx.Response(429, text="Error: Daily quota exceeded per day.", request=req)
    with pytest.raises(QuotaExceededError):
        provider.handle_api_error(httpx.HTTPStatusError("Quota error", request=req, response=resp_quota))

    # 502 / 503 Service Unavailable
    resp_503 = httpx.Response(503, text="Service Unavailable. retryDelay: 20s", request=req)
    with pytest.raises(ServiceUnavailableError) as exc_503:
        provider.handle_api_error(httpx.HTTPStatusError("Unavailable", request=req, response=resp_503))
    assert exc_503.value.retry_delay == 20.0

    # 504 Gateway Timeout
    resp_504 = httpx.Response(504, text="Gateway Timeout", request=req)
    with pytest.raises(ServiceUnavailableError):
        provider.handle_api_error(httpx.HTTPStatusError("Timeout", request=req, response=resp_504))

    # Content filter / blocked
    resp_filter = httpx.Response(400, text="Request failed: content_filter triggered", request=req)
    with pytest.raises(ContentBlockedError):
        provider.handle_api_error(httpx.HTTPStatusError("Filter", request=req, response=resp_filter))

    # Location unsupported
    resp_geo = httpx.Response(400, text="User location is not supported in this region", request=req)
    with pytest.raises(LocationUnsupportedError):
        provider.handle_api_error(httpx.HTTPStatusError("Geo", request=req, response=resp_geo))

    # Generic HTTP error
    resp_418 = httpx.Response(418, text="I am a teapot", request=req)
    with pytest.raises(CHLLMError):
        provider.handle_api_error(httpx.HTTPStatusError("Teapot", request=req, response=resp_418))


@pytest.mark.asyncio
async def test_check_connection_and_close():
    """Проверка check_connection через /models и закрытия клиента."""
    mock_client = MagicMock()
    mock_models_resp = MagicMock(status_code=200)
    mock_client.get = AsyncMock(return_value=mock_models_resp)
    mock_client.aclose = AsyncMock()
    mock_client.is_closed = False

    provider = OpenAIRouterProvider(
        base_url="http://127.0.0.1:8000/v1",
        client=mock_client,
        is_blg=False,
    )

    ok, msg = await provider.check_connection()
    assert ok is True
    assert "/models OK" in msg

    await provider.close()
    # Так как client был передан извне, _owns_client = False, закрытие внешнего клиента не вызывается
    assert mock_client.aclose.call_count == 0

    # Если клиент собственный (_owns_client = True)
    provider._owns_client = True
    await provider.close()
    mock_client.aclose.assert_called_once()
