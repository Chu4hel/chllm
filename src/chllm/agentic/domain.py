"""Доменные модели и протоколы для мультиагентного взаимодействия (chllm.agentic)."""

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
