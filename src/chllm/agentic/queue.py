"""Очередь задач интерактивного взаимодействия Human-in-the-Loop."""

import asyncio
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from ..parser import RobustLLMParser
from .domain import AgentTask, AgentTaskStatus, AgentTaskType

try:
    from chutils.logger import setup_logger
except ImportError:
    import logging

    def setup_logger(name: str = "chllm") -> Any:
        """Создает резервный логгер, если chutils не установлен."""
        return logging.getLogger(name)


logger = setup_logger("chllm.agentic.queue")


class InteractiveTaskQueue:
    """Универсальная очередь задач Human-in-the-Loop для агентов.

    Позволяет регистрировать задачи, ожидать решение от человека
    через асинхронные Future и валидировать ответы по зарегистрированным Pydantic-схемам.
    """

    def __init__(self) -> None:
        """Инициализирует очередь задач и парсер."""
        self._tasks: dict[str, AgentTask] = {}
        self._futures: dict[str, asyncio.Future[BaseModel | str]] = {}
        self._schema_registry: dict[str, type[BaseModel]] = {}
        self._parser = RobustLLMParser()

        self._on_added_callbacks: list[Callable[[AgentTask], None]] = []
        self._on_resolved_callbacks: list[Callable[[str, Any], None]] = []
        self._on_cancelled_callbacks: list[Callable[[str], None]] = []

    def register_schema(self, schema_name: str, model_cls: type[BaseModel]) -> None:
        """Регистрирует Pydantic-модель для валидации схемы ответа.

        Args:
            schema_name: Имя схемы (например, 'ScoreModel').
            model_cls: Класс модели Pydantic.
        """
        self._schema_registry[schema_name] = model_cls

    def subscribe(
        self,
        on_added: Callable[[AgentTask], None] | None = None,
        on_resolved: Callable[[str, Any], None] | None = None,
        on_cancelled: Callable[[str], None] | None = None,
    ) -> None:
        """Подписывает обработчики на события изменения очереди.

        Args:
            on_added: Функция обратного вызова при добавлении задачи.
            on_resolved: Функция обратного вызова при успешном завершении задачи.
            on_cancelled: Функция обратного вызова при отмене задачи.
        """
        if on_added and on_added not in self._on_added_callbacks:
            self._on_added_callbacks.append(on_added)
        if on_resolved and on_resolved not in self._on_resolved_callbacks:
            self._on_resolved_callbacks.append(on_resolved)
        if on_cancelled and on_cancelled not in self._on_cancelled_callbacks:
            self._on_cancelled_callbacks.append(on_cancelled)

    async def enqueue_task(
        self,
        task_id: str,
        title: str,
        prompt: str,
        schema_name: str | None = None,
        task_type: AgentTaskType = AgentTaskType.GENERAL,
        context: dict[str, Any] | None = None,
    ) -> BaseModel | str:
        """Создает задачу в очереди и блокирует вызывающую корутину до ее разрешения.

        Args:
            task_id: Уникальный ID задачи.
            title: Отображаемый заголовок задачи.
            prompt: Текст промпта/запроса для человека.
            schema_name: Имя схемы для валидации результата.
            task_type: Тип задачи.
            context: Дополнительный контекст.

        Returns:
            Результат выполнения задачи (Pydantic модель или строка).
        """
        task = AgentTask(
            id=task_id,
            title=title,
            prompt=prompt,
            schema_name=schema_name,
            task_type=task_type,
            status=AgentTaskStatus.PENDING,
            context=context or {},
        )
        self._tasks[task_id] = task

        loop = asyncio.get_running_loop()
        future: asyncio.Future[BaseModel | str] = loop.create_future()
        self._futures[task_id] = future

        logger.info("Задача %s поставлена в интерактивную очередь: '%s'", task_id, title)

        for cb in self._on_added_callbacks:
            try:
                cb(task)
            except Exception as e:
                logger.warning("Ошибка в callback on_added очереди задач: %s", e)

        return await future

    def resolve_task(self, task_id: str, raw_response: str) -> BaseModel | str:
        """Применяет ответ пользователя/внешнего агента, парсит его и завершает Future.

        Args:
            task_id: ID завершаемой задачи.
            raw_response: Сырой текст ответа.

        Returns:
            Распарсенный объект модели или строка.

        Raises:
            KeyError: Если задача не найдена в очереди.
            ValueError: Если ответ не соответствует ожидаемой схеме.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Задача {task_id} не найдена в интерактивной очереди.")

        task = self._tasks[task_id]
        parsed_result: BaseModel | str = raw_response

        if task.schema_name and task.schema_name in self._schema_registry:
            target_cls = self._schema_registry[task.schema_name]
            clean_json = self._clean_llm_json(raw_response)
            try:
                parsed_result = target_cls.model_validate_json(clean_json)
            except Exception as err:
                try:
                    parsed_dict = self._parser.parse_raw(clean_json)
                    parsed_result = target_cls.model_validate(parsed_dict)
                except Exception as inner_e:
                    logger.error(
                        "Ошибка валидации ответа для задачи %s по схеме %s: %s (json_err: %s)",
                        task_id,
                        task.schema_name,
                        inner_e,
                        err,
                    )
                    raise ValueError(f"Не удалось распарсить ответ по схеме {task.schema_name}: {inner_e}") from inner_e

        task.status = AgentTaskStatus.RESOLVED

        fut = self._futures.get(task_id)
        if fut and not fut.done():
            fut.set_result(parsed_result)

        logger.info("Задача %s успешно разрешена", task_id)

        for cb in self._on_resolved_callbacks:
            try:
                cb(task_id, parsed_result)
            except Exception as e:
                logger.warning("Ошибка в callback on_resolved: %s", e)

        return parsed_result

    def cancel_task(self, task_id: str) -> None:
        """Отменяет выполнение задачи.

        Args:
            task_id: ID отменяемой задачи.
        """
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task.status = AgentTaskStatus.CANCELLED

        fut = self._futures.get(task_id)
        if fut and not fut.done():
            fut.cancel()

        logger.info("Задача %s отменена", task_id)

        for cb in self._on_cancelled_callbacks:
            try:
                cb(task_id)
            except Exception as e:
                logger.warning("Ошибка в callback on_cancelled: %s", e)

    def get_task(self, task_id: str) -> AgentTask | None:
        """Возвращает задачу по ее идентификатору.

        Args:
            task_id: ID задачи.

        Returns:
            Экземпляр AgentTask или None.
        """
        return self._tasks.get(task_id)

    def list_pending_tasks(self) -> list[AgentTask]:
        """Возвращает список всех активных задач в статусе PENDING.

        Returns:
            Список задач.
        """
        return [t for t in self._tasks.values() if t.status == AgentTaskStatus.PENDING]

    def has_pending_tasks(self) -> bool:
        """Проверяет, есть ли ожидающие задачи.

        Returns:
            True, если есть активные задачи, иначе False.
        """
        return any(t.status == AgentTaskStatus.PENDING for t in self._tasks.values())

    def _clean_llm_json(self, raw_text: str) -> str:
        """Очищает Markdown-разметку кода вокруг JSON.

        Args:
            raw_text: Исходный текст ответа.

        Returns:
            Очищенная строка JSON.
        """
        clean = raw_text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean, re.IGNORECASE)
        if match:
            clean = match.group(1).strip()
        return clean
