from unittest.mock import AsyncMock, MagicMock

import pytest

from chllm.orchestrator import Orchestrator


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.execute = AsyncMock()
    # Реализуем простейшую упаковку для теста
    provider.pack_single_prompt = lambda p: {"packed": p}
    return provider


@pytest.mark.asyncio
async def test_execute_single_success(mock_provider):
    """Проверка успешного одиночного запроса."""
    orchestrator = Orchestrator(mock_provider)

    # Имитируем ответ провайдера (объект с полем text)
    mock_response = MagicMock()
    mock_response.text = "AI Response"
    mock_provider.execute.return_value = mock_response

    result = await orchestrator.execute_single("Hello AI")

    assert result == "AI Response"

    # Проверяем, что провайдеру ушли упакованные данные
    mock_provider.execute.assert_called_once_with({"packed": "Hello AI"})


@pytest.mark.asyncio
async def test_execute_single_with_batch_response(mock_provider):
    """Проверка извлечения из батча в ответе."""
    orchestrator = Orchestrator(mock_provider)

    # Имитируем батч-ответ
    mock_item = MagicMock()
    mock_item.translated_text = "Batch Response"
    mock_response = MagicMock()
    mock_response.batch = [mock_item]
    mock_provider.execute.return_value = mock_response

    result = await orchestrator.execute_single("Hello")
    assert result == "Batch Response"


@pytest.mark.asyncio
async def test_execute_single_with_dict_and_generic_container(mock_provider):
    """Проверка извлечения текста из словарей с универсальными полями items и content."""
    orchestrator = Orchestrator(mock_provider)

    mock_provider.execute.return_value = {"items": [{"content": "Generic Response"}]}

    result = await orchestrator.execute_single("Hello")
    assert result == "Generic Response"


@pytest.mark.asyncio
async def test_execute_single_with_dict_direct_result(mock_provider):
    """Проверка извлечения текста из верхнеуровневого словаря с полем output."""
    orchestrator = Orchestrator(mock_provider)

    mock_provider.execute.return_value = {"output": "Direct Output"}

    result = await orchestrator.execute_single("Hello")
    assert result == "Direct Output"
