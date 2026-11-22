"""Провайдер LLM для Orchestrator на базе Zero-API файлового буфера."""

import uuid
from typing import Any

from .queue import FileTaskQueue


class BufferProvider:
    """Провайдер LLM, взаимодействующий через Zero-API файловый буфер (.agent_inbox / .agent_outbox).

    Позволяет использовать Orchestrator и AgentOrchestrator без прямого доступа
    к сети, взаимодействуя с внешним ассистентом или IDE через файлы.
    """

    def __init__(
        self,
        file_queue: FileTaskQueue,
        task_type: str = "GENERAL",
        timeout: float = 300.0,
        poll_interval: float = 1.0,
    ) -> None:
        """Инициализирует провайдер файлового буфера.

        Args:
            file_queue: Экземпляр очереди FileTaskQueue.
            task_type: Тип задачи по умолчанию.
            timeout: Максимальное время ожидания ответа в секундах.
            poll_interval: Интервал проверки появления ответа в секундах.
        """
        self.file_queue = file_queue
        self.task_type = task_type
        self.timeout = timeout
        self.poll_interval = poll_interval

    def pack_single_prompt(self, prompt: str) -> dict[str, Any]:
        """Упаковывает одиночный промпт в структуру данных задачи.

        Args:
            prompt: Текст промпта.

        Returns:
            Словарь с данными задачи.
        """
        task_id = f"buf_{uuid.uuid4().hex[:8]}"
        return {
            "task_id": task_id,
            "task_type": self.task_type,
            "payload": {"prompt": prompt},
        }

    async def execute(self, data: object) -> str | dict[str, Any]:
        """Выполняет запрос через постановку в очередь и ожидание в outbox.

        Args:
            data: Данные задачи (словарь или объект с task_id и payload).

        Returns:
            Ответ модели (строка или извлеченные данные).

        Raises:
            ValueError: Если входные данные имеют неверный формат.
            TimeoutError: При превышении таймаута ожидания.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Ожидался словарь данных для BufferProvider, получено: {type(data)}")

        task_id = str(data.get("task_id") or f"buf_{uuid.uuid4().hex[:8]}")
        task_type = str(data.get("task_type") or self.task_type)
        payload = data.get("payload")
        if payload is None:
            payload = data

        self.file_queue.push_task(
            task_id=task_id,
            task_type=task_type,
            payload=payload if isinstance(payload, dict) else {"prompt": str(payload)},
        )

        res = await self.file_queue.wait_for_result(
            task_id=task_id,
            timeout=self.timeout,
            poll_interval=self.poll_interval,
        )

        if isinstance(res, dict):
            for key in ("result", "answer", "content", "response", "text"):
                if key in res:
                    val = res[key]
                    if isinstance(val, str):
                        return val
                    return res
        return res
