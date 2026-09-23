from unittest.mock import AsyncMock, MagicMock

import pytest

from chllm.providers import GenericCallableProvider, OpenAICompatibleProvider


@pytest.mark.asyncio
async def test_generic_callable_provider():
    """Тест GenericCallableProvider с асинхронной лямбдой / функцией."""

    async def fake_llm(data):
        return f"Echo: {data}"

    provider = GenericCallableProvider(fake_llm)
    packed = provider.pack_single_prompt("Test prompt")
    assert packed == "Test prompt"

    result = await provider.execute("Hello")
    assert result == "Echo: Hello"


@pytest.mark.asyncio
async def test_openai_compatible_provider():
    """Тест OpenAICompatibleProvider с моком OpenAI клиента."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "OpenAI answer"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = OpenAICompatibleProvider(
        client=mock_client,
        model="gpt-4o-mini",
        temperature=0.7,
    )

    packed = provider.pack_single_prompt("My question")
    assert packed == [{"role": "user", "content": "My question"}]

    result = await provider.execute(packed)
    assert result == "OpenAI answer"

    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "My question"}],
        temperature=0.7,
    )
