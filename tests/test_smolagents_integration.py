"""Тесты интеграции со smolagents."""

import asyncio
import threading

import pytest

import chllm.integrations.smolagents as smol_integration
from chllm.integrations.smolagents.tools import BaseSmolToolAdapter


def test_base_smol_tool_adapter_dispatch() -> None:
    """Проверяет потокобезопасный вызов корутины через BaseSmolToolAdapter."""
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    try:

        async def sample_coro(value: int) -> int:
            await asyncio.sleep(0.01)
            return value * 2

        result = BaseSmolToolAdapter.dispatch_coroutine(
            coro=sample_coro(21),
            loop=loop,
            timeout=2.0,
        )
        assert result == 42
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1.0)


def test_smolagents_import_error_when_not_installed() -> None:
    """Проверяет выброс понятного ImportError при обращении к моделям без smolagents."""
    if not smol_integration._HAS_SMOLAGENTS:
        with pytest.raises(ImportError) as exc_info:
            _ = smol_integration.WebChatSmolModel
        assert "chllm[smolagents]" in str(exc_info.value)
