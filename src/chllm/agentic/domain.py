"""Доменные модели и протоколы для мультиагентного взаимодействия (chllm.agentic)."""

from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ForkMode(str, Enum):
    """Режим инициализации контекста дочерней агентной сессии."""

    ISOLATED = "isolated"
    INHERIT_PROMPT = "inherit_prompt"
    FORK_HISTORY = "fork_history"


class AgentMessage(BaseModel):
    """Сообщение между агентными сессиями (Mailbox payload).

    Attributes:
        sender_id: Идентификатор сессии отправителя.
        recipient_id: Идентификатор сессии получателя.
        content: Текст сообщения или полезная нагрузка.
        timestamp: Время создания сообщения.
        metadata: Дополнительные метаданные сообщения.
    """

    model_config = ConfigDict(frozen=True)

    sender_id: str = Field(..., description="ID сессии отправителя")
    recipient_id: str = Field(..., description="ID сессии получателя")
    content: str = Field(..., description="Текст сообщения или полезная нагрузка")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Время отправки")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Метаданные")


class SessionSnapshot(BaseModel):
    """Снимок состояния сессии для сериализации и хранения.

    Attributes:
        session_id: Уникальный идентификатор сессии.
        parent_id: Идентификатор родительской сессии (если есть).
        role: Роль или специфика агента (например, "LoreExpert").
        system_prompt: Базовый системный промпт сессии.
        history: История сообщений диалога [{"role": "user/assistant", "content": "..."}].
        created_at: Время создания сессии.
        metadata: Пользовательские метаданные.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = Field(..., description="Уникальный ID сессии")
    parent_id: str | None = Field(default=None, description="ID родительской сессии")
    role: str = Field(default="Assistant", description="Роль агента")
    system_prompt: str = Field(default="", description="Системный промпт")
    history: list[dict[str, str]] = Field(default_factory=list, description="История диалога")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Время создания")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Метаданные сессии")


class ISessionStorage(Protocol):
    """Интерфейс хранилища сессий (Dependency Inversion)."""

    async def save_session(self, snapshot: SessionSnapshot) -> None:
        """Сохраняет или обновляет снимок сессии.

        Args:
            snapshot: Экземпляр SessionSnapshot.
        """
        ...

    async def get_session(self, session_id: str) -> SessionSnapshot | None:
        """Возвращает снимок сессии по ее идентификатору.

        Args:
            session_id: Идентификатор сессии.

        Returns:
            Экземпляр SessionSnapshot или None, если сессия не найдена.
        """
        ...

    async def list_sessions(self, parent_id: str | None = None) -> list[SessionSnapshot]:
        """Возвращает список сессий, опционально фильтруя по родителю.

        Args:
            parent_id: Опциональный ID родительской сессии.

        Returns:
            Список снимков сессий.
        """
        ...

    async def delete_session(self, session_id: str) -> bool:
        """Удаляет сессию из хранилища.

        Args:
            session_id: Идентификатор удаляемой сессии.

        Returns:
            True, если сессия была найдена и удалена, иначе False.
        """
        ...


class AgentTaskStatus(str, Enum):
    """Статусы выполнения задачи агентом."""

    PENDING = "pending"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class AgentMode(str, Enum):
    """Протокол взаимодействия агента с внешним миром / оператором."""

    API = "api"  # Прямой вызов LLM через API (OpenAI, Anthropic, vLLM, Ollama)
    WEB_CHAT = "web_chat"  # Интерактивный веб-чат Human-in-the-Loop
    BUFFER = "buffer"  # Zero-API файловый буфер (.agent_inbox / .agent_outbox)


class AgentTaskType(str, Enum):
    """Универсальные типы задач агентского взаимодействия."""

    SELF_HEALING = "self_healing"
    CODE_FIX = "code_fix"
    ASSISTANT = "assistant"
    SCORE = "score"
    REPLY = "reply"
    GENERAL = "general"


class AgentTask(BaseModel):
    """Модель задачи агента в интерактивной очереди или буфере."""

    id: str = Field(..., description="Уникальный идентификатор задачи.")
    title: str = Field(..., description="Заголовок задачи для UI / логов.")
    prompt: str = Field(..., description="Текст промпта или инструкция.")
    schema_name: str | None = Field(
        default=None,
        description="Имя зарегистрированной Pydantic-схемы для парсинга и валидации.",
    )
    task_type: AgentTaskType = Field(
        default=AgentTaskType.GENERAL,
        description="Категория/тип задачи.",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Время создания задачи.",
    )
    status: AgentTaskStatus = Field(
        default=AgentTaskStatus.PENDING,
        description="Текущий статус задачи.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Произвольный контекст задачи (метаданные, аргументы, история).",
    )


class AgentStepResult(BaseModel):
    """Результат выполнения одного шага рассуждения или вызова инструмента."""

    step_index: int = Field(..., description="Номер шага агента.")
    raw_response: str | None = Field(
        default=None,
        description="Сырой ответ языковой модели.",
    )
    parsed_data: dict[str, Any] | None = Field(
        default=None,
        description="Структурированные данные после парсинга.",
    )
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Список вызванных инструментов и аргументов.",
    )
    error: str | None = Field(
        default=None,
        description="Текст ошибки, если шаг завершился сбоем.",
    )


class IInteractiveQueue(Protocol):
    """Протокол интерактивной очереди задач для Human-in-the-Loop."""

    def register_schema(self, schema_name: str, model_cls: type[BaseModel]) -> None:
        """Регистрирует Pydantic-модель для валидации ответа.

        Args:
            schema_name: Имя схемы.
            model_cls: Класс модели Pydantic.
        """
        ...

    async def enqueue_task(
        self,
        task_id: str,
        title: str,
        prompt: str,
        schema_name: str | None = None,
        task_type: AgentTaskType = AgentTaskType.GENERAL,
        context: dict[str, Any] | None = None,
    ) -> BaseModel | str:
        """Помещает задачу в очередь и ожидает решения от человека.

        Args:
            task_id: Идентификатор задачи.
            title: Заголовок задачи.
            prompt: Текст промпта/инструкции.
            schema_name: Имя схемы ответа.
            task_type: Тип задачи.
            context: Дополнительный контекст.

        Returns:
            Результат выполнения задачи.
        """
        ...

    def resolve_task(self, task_id: str, raw_response: str) -> BaseModel | str:
        """Завершает задачу полученным текстовым ответом с парсингом.

        Args:
            task_id: Идентификатор задачи.
            raw_response: Сырой текст ответа от пользователя или LLM.

        Returns:
            Распарсенный или исходный результат.
        """
        ...

    def cancel_task(self, task_id: str) -> None:
        """Отменяет задачу в очереди.

        Args:
            task_id: Идентификатор задачи.
        """
        ...

    def subscribe(
        self,
        on_added: Callable[[AgentTask], None] | None = None,
        on_resolved: Callable[[str, Any], None] | None = None,
        on_cancelled: Callable[[str], None] | None = None,
    ) -> None:
        """Подписывает обработчики на события изменения очереди.

        Args:
            on_added: Коллбэк добавления задачи.
            on_resolved: Коллбэк завершения задачи.
            on_cancelled: Коллбэк отмены задачи.
        """
        ...
