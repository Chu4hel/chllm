"""Универсальный провайдер, оборачивающий асинхронную функцию."""

from collections.abc import Awaitable, Callable
from typing import Any


class GenericCallableProvider:
    """Универсальный провайдер, оборачивающий любую асинхронную функцию.

    Позволяет быстро интегрировать любую кастомную логику вызова LLM в Orchestrator.
    """

    def __init__(self, async_callable: Callable[[Any], Awaitable[Any]]) -> None:
        """Инициализирует провайдер.

        Args:
            async_callable: Асинхронная функция, принимающая данные запроса и возвращающая ответ.
        """
        self._async_callable = async_callable

    async def execute(self, data: object) -> object:
        """Выполняет запрос через переданную асинхронную функцию.

        Args:
            data: Данные запроса.

        Returns:
            Ответ от асинхронной функции.
        """
        return await self._async_callable(data)

    def pack_single_prompt(self, prompt: str) -> object:
        """По умолчанию передает строку промпта как есть.

        Args:
            prompt: Текст промпта.

        Returns:
            Текст промпта без изменений.
        """
        return prompt
