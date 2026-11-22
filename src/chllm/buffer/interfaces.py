"""Интерфейсы и протоколы файлового буфера задач (Zero-API)."""

from pathlib import Path
from typing import Any, Protocol


class ITaskQueue(Protocol):
    """Протокол файлового или удаленного буфера задач агента."""

    def push_task(self, task_id: str, task_type: str, payload: dict[str, Any]) -> Path:
        """Помещает задачу в очередь задач.

        Args:
            task_id: Уникальный ID задачи.
            task_type: Тип задачи.
            payload: Данные задачи.

        Returns:
            Путь к созданному артефакту/файлу задачи.
        """
        ...

    def get_result(self, task_id: str) -> dict[str, Any] | None:
        """Получает готовый результат задачи, если он готов.

        Args:
            task_id: Уникальный ID задачи.

        Returns:
            Словарь с результатом или None.
        """
        ...

    async def wait_for_result(
        self,
        task_id: str,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> dict[str, Any]:
        """Ожидает появление результата задачи в асинхронном режиме.

        Args:
            task_id: Уникальный ID задачи.
            timeout: Таймаут ожидания в секундах.
            poll_interval: Интервал проверки.

        Returns:
            Словарь с результатом выполнения задачи.
        """
        ...
