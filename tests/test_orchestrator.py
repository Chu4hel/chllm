from unittest.mock import AsyncMock

import pytest

from chllm.exceptions import ContentBlockedError, RateLimitError
from chllm.orchestrator import Orchestrator, RetryStrategy


class MockProvider:
    def __init__(self):
        self.call_count = 0
        self.execute = AsyncMock()

    def pack_single_prompt(self, prompt: str) -> object:
        return prompt


@pytest.mark.asyncio
@pytest.mark.slow
async def test_orchestrator_retries_on_rate_limit():
    provider = MockProvider()
    # Сначала ошибка, потом успех
    provider.execute.side_effect = [RateLimitError("Limit"), "Success"]

    orchestrator = Orchestrator(provider)
    result = await orchestrator.execute("data")

    assert result == "Success"
    assert provider.execute.call_count == 2


@pytest.mark.asyncio
async def test_orchestrator_splits_on_blocked_content():
    provider = MockProvider()

    # Провайдер блокирует батч целиком, но принимает одиночные элементы
    async def mock_execute(data):
        if isinstance(data, list) and len(data) > 1:
            raise ContentBlockedError("Safety")
        return f"Processed {data}"

    provider.execute.side_effect = mock_execute

    orchestrator = Orchestrator(provider)
    batch = ["item1", "item2"]

    # Ожидаем, что оркестратор поделит батч и вернет список результатов
    result = await orchestrator.execute(batch)

    assert result == ["Processed ['item1']", "Processed ['item2']"]
    # 1 вызов (бабах) + 2 вызова (половинки)
    assert provider.execute.call_count == 3


@pytest.mark.asyncio
@pytest.mark.slow
async def test_orchestrator_fails_after_max_attempts():
    provider = MockProvider()
    provider.execute.side_effect = RateLimitError("Limit")

    orchestrator = Orchestrator(provider, strategy=RetryStrategy(max_retries=2))

    with pytest.raises(RateLimitError):
        await orchestrator.execute("data")

    assert provider.execute.call_count == 3  # 1 попытка + 2 ретрая


@pytest.mark.asyncio
async def test_orchestrator_custom_strategy():
    provider = MockProvider()
    provider.execute.side_effect = [ValueError("Custom"), "Success"]

    # Настраиваем стратегию на ретрай ValueError
    strategy = RetryStrategy(max_retries=1, retryable_errors=(ValueError,), base_delay=0.1)
    orchestrator = Orchestrator(provider, strategy=strategy)

    result = await orchestrator.execute("data")
    assert result == "Success"
    assert provider.execute.call_count == 2
