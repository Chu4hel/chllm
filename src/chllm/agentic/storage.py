"""Хранилища сессий (InMemorySessionStorage)."""

from .domain import ISessionStorage, SessionSnapshot


class InMemorySessionStorage(ISessionStorage):
    """Хранилище сессий в оперативной памяти."""

    def __init__(self) -> None:
        """Инициализирует пустое in-memory хранилище."""
        self._sessions: dict[str, SessionSnapshot] = {}

    async def save_session(self, snapshot: SessionSnapshot) -> None:
        """Сохраняет снимок сессии в памяти.

        Args:
            snapshot: Снимок сессии.
        """
        self._sessions[snapshot.session_id] = snapshot

    async def get_session(self, session_id: str) -> SessionSnapshot | None:
        """Извлекает сессию по ID.

        Args:
            session_id: ID сессии.

        Returns:
            Снимок сессии или None.
        """
        return self._sessions.get(session_id)

    async def list_sessions(self, parent_id: str | None = None) -> list[SessionSnapshot]:
        """Возвращает список всех сессий или дочерних для указанного родителя.

        Args:
            parent_id: Опциональный ID родительской сессии.

        Returns:
            Список снимков сессий.
        """
        if parent_id is not None:
            return [s for s in self._sessions.values() if s.parent_id == parent_id]
        return list(self._sessions.values())

    async def delete_session(self, session_id: str) -> bool:
        """Удаляет сессию из памяти.

        Args:
            session_id: ID сессии.

        Returns:
            True, если сессия существовала и была удалена, иначе False.
        """
        return self._sessions.pop(session_id, None) is not None
