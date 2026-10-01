"""Пул агентных сессий (ConversationPool) и координация взаимодействия."""

import uuid
from collections.abc import Callable
from typing import Any

from ..exceptions import SessionNotFoundError
from ..orchestrator import AgentOrchestrator
from ..skills.registry import SkillRegistry
from .bus import SessionMessageBus
from .domain import AgentMessage, ForkMode, ISessionStorage
from .session import AgentSession
from .storage import InMemorySessionStorage


class ConversationPool:
    """Пул для управления жизненным циклом агентных сессий и шиной сообщений."""

    def __init__(
        self,
        orchestrator: AgentOrchestrator | Callable[[], AgentOrchestrator],
        storage: ISessionStorage | None = None,
        bus: SessionMessageBus | None = None,
        default_skills: SkillRegistry | None = None,
        default_tools: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        """Инициализирует пул сессий.

        Args:
            orchestrator: Экземпляр оркестратора или фабрика для его создания.
            storage: Хранилище снимков сессий (по умолчанию InMemorySessionStorage).
            bus: Шина сообщений между сессиями (по умолчанию SessionMessageBus).
            default_skills: Реестр навыков по умолчанию для всех сессий.
            default_tools: Инструменты по умолчанию для всех сессий.
        """
        self._orchestrator_factory = (
            orchestrator
            if callable(orchestrator) and not isinstance(orchestrator, AgentOrchestrator)
            else lambda: orchestrator
        )
        self.storage: ISessionStorage = storage or InMemorySessionStorage()
        self.bus: SessionMessageBus = bus or SessionMessageBus()
        self.default_skills = default_skills
        self.default_tools: dict[str, Callable[..., Any]] = dict(default_tools or {})
        self._active_sessions: dict[str, AgentSession] = {}

    def _generate_id(self, prefix: str = "session") -> str:
        """Генерирует уникальный идентификатор сессии."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def create_session(
        self,
        session_id: str | None = None,
        parent_id: str | None = None,
        role: str = "Assistant",
        system_prompt: str = "",
        tools: dict[str, Callable[..., Any]] | None = None,
        skills: SkillRegistry | None = None,
        metadata: dict[str, Any] | None = None,
        enable_agentic_tools: bool = True,
    ) -> AgentSession:
        """Создает новую изолированную сессию агента.

        Args:
            session_id: Опциональный явный ID сессии (иначе генерируется автоматически).
            parent_id: ID родительской сессии (если есть).
            role: Роль агента в сессии.
            system_prompt: Системная инструкция агента.
            tools: Пользовательские инструменты.
            skills: Реестр навыков (если не передан, берется default_skills).
            metadata: Пользовательские метаданные.
            enable_agentic_tools: Добавлять ли встроенные инструменты delegate_task, send_message.

        Returns:
            Созданный экземпляр AgentSession.
        """
        sid = session_id or self._generate_id(prefix=role.lower())
        active_tools = dict(self.default_tools)
        if tools:
            active_tools.update(tools)

        if enable_agentic_tools:
            active_tools.update(self._build_agentic_tools(sid))

        session = AgentSession(
            session_id=sid,
            orchestrator=self._orchestrator_factory(),
            parent_id=parent_id,
            role=role,
            system_prompt=system_prompt,
            tools=active_tools,
            skills=skills or self.default_skills,
            history=[],
            metadata=metadata or {},
        )

        self._active_sessions[sid] = session
        return session

    def fork_session(
        self,
        parent_session_id: str,
        new_session_id: str | None = None,
        mode: ForkMode = ForkMode.ISOLATED,
        new_role: str | None = None,
        system_prompt: str | None = None,
        tools: dict[str, Callable[..., Any]] | None = None,
    ) -> AgentSession:
        """Создает дочернюю сессию на основе родительской с выбранным режимом форка.

        Args:
            parent_session_id: ID родительской сессии.
            new_session_id: Опциональный ID для новой сессии.
            mode: Режим форка (ISOLATED, INHERIT_PROMPT, FORK_HISTORY).
            new_role: Новая роль субагента (по умолчанию наследует роль родителя с префиксом Sub).
            system_prompt: Новая или дополнительная системная инструкция.
            tools: Дополнительные инструменты.

        Returns:
            Созданная дочерняя сессия.

        Raises:
            SessionNotFoundError: Если родительская сессия не найдена.
        """
        parent = self._active_sessions.get(parent_session_id)
        if parent is None:
            raise SessionNotFoundError(f"Родительская сессия '{parent_session_id}' не найдена")

        role = new_role or f"Sub_{parent.role}"
        history: list[dict[str, str]] = []
        final_prompt = system_prompt or ""

        if mode == ForkMode.INHERIT_PROMPT:
            combined = [parent.system_prompt]
            if system_prompt:
                combined.append(system_prompt)
            final_prompt = "\n\n".join([p for p in combined if p])
        elif mode == ForkMode.FORK_HISTORY:
            combined = [parent.system_prompt]
            if system_prompt:
                combined.append(system_prompt)
            final_prompt = "\n\n".join([p for p in combined if p])
            history = [dict(msg) for msg in parent.history]

        sid = new_session_id or self._generate_id(prefix=role.lower())
        merged_tools = dict(parent.tools)
        if tools:
            merged_tools.update(tools)
        merged_tools.update(self._build_agentic_tools(sid))

        child = AgentSession(
            session_id=sid,
            orchestrator=self._orchestrator_factory(),
            parent_id=parent_session_id,
            role=role,
            system_prompt=final_prompt,
            tools=merged_tools,
            skills=parent.skills,
            history=history,
            metadata={"fork_mode": mode.value},
        )

        self._active_sessions[sid] = child
        return child

    def get_session(self, session_id: str) -> AgentSession | None:
        """Возвращает активную сессию по ID.

        Args:
            session_id: ID сессии.

        Returns:
            Экземпляр AgentSession или None.
        """
        return self._active_sessions.get(session_id)

    async def delegate(
        self,
        parent_id: str,
        role: str,
        task: str,
        mode: ForkMode = ForkMode.ISOLATED,
        system_prompt: str = "",
        tools: dict[str, Callable[..., Any]] | None = None,
        max_iterations: int = 5,
    ) -> str:
        """Синхронно делегирует подзадачу новой дочерней сессии и возвращает результат.

        Args:
            parent_id: ID родительской сессии.
            role: Роль субагента (например, "LoreExpert").
            task: Текст задачи.
            mode: Режим контекста форка (по умолчанию ISOLATED).
            system_prompt: Системная инструкция для субагента.
            tools: Дополнительные инструменты субагента.
            max_iterations: Максимальное число итераций выполнения.

        Returns:
            Чистый результат выполнения подзадачи.
        """
        subsession = self.fork_session(
            parent_session_id=parent_id,
            mode=mode,
            new_role=role,
            system_prompt=system_prompt,
            tools=tools,
        )

        try:
            result = await subsession.execute(prompt=task, max_iterations=max_iterations)
            await self.storage.save_session(subsession.to_snapshot())
            return result
        except Exception as e:
            return f"Subagent execution failed: {type(e).__name__}: {e}"

    def _build_agentic_tools(self, current_session_id: str) -> dict[str, Callable[..., Any]]:
        """Создает набор встроенных инструментов для LLM текущей сессии."""

        async def delegate_task(role: str, task: str, mode: str = "isolated") -> str:
            """Делегирует подзадачу специализированному субагенту и возвращает чистый результат."""
            try:
                fork_mode = ForkMode(mode.lower().strip())
            except ValueError:
                fork_mode = ForkMode.ISOLATED

            return await self.delegate(
                parent_id=current_session_id,
                role=role,
                task=task,
                mode=fork_mode,
            )

        async def send_message(recipient_id: str, content: str) -> str:
            """Отправляет асинхронное сообщение другой агентной сессии через шину."""
            msg = AgentMessage(
                sender_id=current_session_id,
                recipient_id=recipient_id.strip(),
                content=content.strip(),
            )
            await self.bus.send_message(msg)
            return f"Message sent to {recipient_id}."

        def read_inbox() -> str:
            """Читает все накопленные входящие сообщения из почтового ящика."""
            messages = self.bus.read_inbox(current_session_id)
            if not messages:
                return "Inbox is empty."
            formatted: list[str] = [f"Inbox ({len(messages)} messages):"]
            for m in messages:
                formatted.append(f"- From {m.sender_id}: {m.content}")
            return "\n".join(formatted)

        return {
            "delegate_task": delegate_task,
            "send_message": send_message,
            "read_inbox": read_inbox,
        }
