from unittest.mock import AsyncMock, MagicMock

import pytest

from chllm.orchestrator import AgentOrchestrator


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.execute = AsyncMock()
    provider.pack_single_prompt = lambda p: p
    return provider


@pytest.mark.asyncio
async def test_agent_orchestrator_loop_with_tool(mock_provider):
    """Тест цикла агента: вызов инструмента -> ответ инструмента -> финал."""
    # Настраиваем последовательность ответов ИИ
    # 1. ИИ просит вызвать инструмент (JSON)
    # 2. ИИ дает финальный ответ (Текст)
    mock_provider.execute.side_effect = [
        '{"name": "get_weather", "args": {"city": "Moscow"}}',
        "The weather in Moscow is 20C.",
    ]

    # Фейковый инструмент
    async def get_weather(city: str):
        return f"Weather data for {city}: 20C"

    orchestrator = AgentOrchestrator(mock_provider)
    tools = {"get_weather": get_weather}

    result = await orchestrator.execute_loop("What is the weather?", tools=tools)

    assert result == "The weather in Moscow is 20C."
    # Проверяем, что было 2 вызова провайдера
    assert mock_provider.execute.call_count == 2


@pytest.mark.asyncio
async def test_agent_orchestrator_max_iterations(mock_provider):
    """Тест прерывания по лимиту итераций."""
    # ИИ бесконечно просит инструмент
    mock_provider.execute.return_value = '{"name": "loop", "args": {}}'

    orchestrator = AgentOrchestrator(mock_provider)

    # Ограничиваем 2 итерациями
    result = await orchestrator.execute_loop("Loop me", tools={"loop": AsyncMock()}, max_iterations=2)

    assert mock_provider.execute.call_count == 2
    # Возвращается последний ответ перед прерыванием (или сообщение о лимите)
    assert "loop" in result
