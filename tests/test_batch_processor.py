import asyncio

import pytest

from chllm.batch_processor import AsyncBatchProcessor


@pytest.mark.asyncio
async def test_async_batch_processor_success():
    """Тест успешной параллельной обработки элементов."""
    processor = AsyncBatchProcessor(concurrency_limit=2)

    async def double(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 2

    items = [1, 2, 3, 4, 5]
    results = await processor.process(items, double)

    assert len(results) == 5
    for i, res in enumerate(results):
        assert res.success is True
        assert res.index == i
        assert res.input_item == items[i]
        assert res.output == items[i] * 2
        assert res.error is None


@pytest.mark.asyncio
async def test_async_batch_processor_handles_errors():
    """Тест изоляции ошибок в элементах при return_exceptions=True."""
    processor = AsyncBatchProcessor(concurrency_limit=3)

    async def flaky_task(x: int) -> str:
        if x == 2:
            raise ValueError("Bad number 2")
        return f"OK {x}"

    items = [1, 2, 3]
    results = await processor.process(items, flaky_task, return_exceptions=True)

    assert len(results) == 3
    assert results[0].success is True
    assert results[0].output == "OK 1"

    assert results[1].success is False
    assert isinstance(results[1].error, ValueError)
    assert "Bad number 2" in str(results[1].error)

    assert results[2].success is True
    assert results[2].output == "OK 3"


@pytest.mark.asyncio
async def test_async_batch_processor_empty():
    """Тест пустого списка элементов."""
    processor = AsyncBatchProcessor()

    async def identity(x: str) -> str:
        return x

    empty_items: list[str] = []
    results = await processor.process(empty_items, identity)
    assert results == []
