import asyncio

import pytest
from pydantic import BaseModel

from chllm.parser import RobustLLMParser


class StreamItem(BaseModel):
    id: int
    name: str


@pytest.mark.asyncio
async def test_parse_stream_yields_as_they_arrive():
    """Тест потокового парсинга по мере поступления чанков."""
    parser = RobustLLMParser()

    # Имитируем поток токенов
    chunks = [
        '{"items": [',
        '{"id": 1, "name": "Item 1"}',
        ",",
        '{"id": 2, "name": "Item 2"}',
        "]}",
    ]

    async def fake_stream():
        for chunk in chunks:
            await asyncio.sleep(0.01)
            yield chunk

    results = []
    async for item in parser.parse_stream(fake_stream(), validation_model=StreamItem):
        results.append(item)

    assert len(results) == 2
    assert isinstance(results[0], StreamItem)
    assert results[0].id == 1
    assert results[0].name == "Item 1"
    assert results[1].id == 2
    assert results[1].name == "Item 2"


@pytest.mark.asyncio
async def test_parse_stream_raw_dicts():
    """Тест потокового парсинга без модели (в словари)."""
    parser = RobustLLMParser()

    chunks = [
        '[{"id": 10}, {"id": 2',
        "0}]",
    ]

    async def fake_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for item in parser.parse_stream(fake_stream()):
        results.append(item)

    assert len(results) == 2
    assert results[0]["id"] == 10
    assert results[1]["id"] == 20
