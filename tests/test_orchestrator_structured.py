from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from chllm.exceptions import ParsingError
from chllm.orchestrator import Orchestrator


class Product(BaseModel):
    title: str
    price: float


@pytest.mark.asyncio
async def test_execute_structured_success_first_try():
    """Тест успешного выполнения execute_structured с первой попытки."""
    mock_provider = MagicMock()
    mock_provider.pack_single_prompt = lambda p: p
    mock_provider.execute = AsyncMock(return_value='{"items": [{"title": "Book", "price": 12.5}]}')

    orchestrator = Orchestrator(mock_provider)
    products = await orchestrator.execute_structured("Give me books", Product)

    assert len(products) == 1
    assert isinstance(products[0], Product)
    assert products[0].title == "Book"
    assert products[0].price == 12.5
    assert mock_provider.execute.call_count == 1


@pytest.mark.asyncio
async def test_execute_structured_self_correction_success():
    """Тест успешного самоисправления после битого первого ответа."""
    mock_provider = MagicMock()
    mock_provider.pack_single_prompt = lambda p: p

    # 1-й ответ: сломанный JSON, 2-й ответ: корректный
    mock_provider.execute = AsyncMock(
        side_effect=[
            '{"items": [{"title": "Broken"',  # сломано
            '{"items": [{"title": "Pen", "price": 1.5}]}',  # исправлено
        ]
    )

    orchestrator = Orchestrator(mock_provider)
    products = await orchestrator.execute_structured(
        "Give me items",
        Product,
        max_correction_retries=2,
    )

    assert len(products) == 1
    assert products[0].title == "Pen"
    assert products[0].price == 1.5
    assert mock_provider.execute.call_count == 2


@pytest.mark.asyncio
async def test_execute_structured_raises_parsing_error_after_max_retries():
    """Тест выбрасывания ParsingError при исчерпании попыток самоисправления."""
    mock_provider = MagicMock()
    mock_provider.pack_single_prompt = lambda p: p
    mock_provider.execute = AsyncMock(return_value="Total gibberish that cannot be parsed")

    orchestrator = Orchestrator(mock_provider)

    with pytest.raises(ParsingError) as exc_info:
        await orchestrator.execute_structured(
            "Prompt",
            Product,
            max_correction_retries=1,
        )

    assert "Не удалось распарсить валидные элементы" in str(exc_info.value)
    assert mock_provider.execute.call_count == 2  # 1 основная + 1 повторная
