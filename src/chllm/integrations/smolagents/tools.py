"""Базовые адаптеры и утилиты для инструментов smolagents."""

import asyncio
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from smolagents import Tool
else:
    Tool = Any

try:
    from chutils.logger import setup_logger
except ImportError:
    import logging

    def setup_logger(name: str = "chllm") -> Any:
        """Создает резервный логгер, если chutils не установлен."""
        return logging.getLogger(name)


logger = setup_logger("chllm.integrations.smolagents.tools")

T = TypeVar("T")


class BaseSmolToolAdapter:
    """Базовый адаптер для построения инструментов smolagents с потокобезопасным вызовом."""

    @staticmethod
    def dispatch_coroutine(
        coro: Coroutine[Any, Any, T],
        loop: asyncio.AbstractEventLoop,
        timeout: float = 60.0,
    ) -> T:
        """Выполняет асинхронную корутину в целевом цикле событий и ожидает результат.

        Args:
            coro: Корутина для выполнения.
            loop: Активный цикл событий asyncio.
            timeout: Максимальное время ожидания выполнения.

        Returns:
            Результат выполнения корутины.

        Raises:
            TimeoutError: При превышении времени ожидания.
        """
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result(timeout=timeout)
