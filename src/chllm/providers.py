"""
Готовые провайдеры-адаптеры для работы с Orchestrator.
Обеспечивают интеграцию с произвольными асинхронными функциями и популярными LLM SDK.
"""

from collections.abc import Awaitable, Callable
from typing import Any


class GenericCallableProvider:
    """Универсальный провайдер, оборачивающий любую асинхронную функцию.

    Позволяет быстро интегрировать любую кастомную логику вызова LLM в Orchestrator.
    """

    def __init__(self, async_callable: Callable[[Any], Awaitable[Any]]):
        """
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


class OpenAICompatibleProvider:
    """Провайдер для любых клиентов, совместимых с OpenAI Chat Completions API.

    Работает с AsyncOpenAI, LiteLLM, Ollama, vLLM, DeepSeek и др.
    Не требует жесткой зависимости от пакета openai в проекте.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float = 0.0,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        """
        Args:
            client: Экземпляр асинхронного клиента (например, AsyncOpenAI).
            model: Имя модели (например, 'gpt-4o-mini', 'deepseek-chat').
            temperature: Температура генерации.
            extra_kwargs: Дополнительные аргументы для client.chat.completions.create.
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs or {}

    async def execute(self, data: object) -> str:
        """Выполняет запрос к OpenAI-совместимому API.

        Args:
            data: Строка, список сообщений или словарь с ключом 'messages'.

        Returns:
            Текстовое содержимое первого ответа модели.
        """
        messages: list[dict[str, str]]
        if isinstance(data, str):
            messages = [{"role": "user", "content": data}]
        elif isinstance(data, list):
            messages = data  # type: ignore[assignment]
        elif isinstance(data, dict) and "messages" in data:
            messages = data["messages"]
        else:
            messages = [{"role": "user", "content": str(data)}]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.extra_kwargs,
        )

        # Извлекаем текст ответа
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return choice.message.content or ""
        return str(response)

    def pack_single_prompt(self, prompt: str) -> object:
        """Упаковывает промпт в структуру сообщений Chat API.

        Args:
            prompt: Текст промпта.

        Returns:
            Список со структурой сообщений для OpenAI Chat API.
        """
        return [{"role": "user", "content": prompt}]
