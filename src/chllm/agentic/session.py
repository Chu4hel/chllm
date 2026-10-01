"""Изолированная агентная сессия (AgentSession)."""

from collections.abc import Callable
from typing import Any

from ..orchestrator import AgentOrchestrator
from ..skills.registry import SkillRegistry
from .domain import SessionSnapshot


class AgentSession:
    """Изолированная сессия агента со своей историей, инструкциями и инструментами."""

    def __init__(
        self,
        session_id: str,
        orchestrator: AgentOrchestrator,
        parent_id: str | None = None,
        role: str = "Assistant",
        system_prompt: str = "",
        tools: dict[str, Callable[..., Any]] | None = None,
        skills: SkillRegistry | None = None,
        history: list[dict[str, str]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Инициализирует агентную сессию.

        Args:
            session_id: Уникальный ID сессии.
            orchestrator: Экземпляр AgentOrchestrator для выполнения LLM-цикла.
            parent_id: ID родительской сессии (если есть).
            role: Роль агента в сессии.
            system_prompt: Системная инструкция агента.
            tools: Словарь доступных Python-инструментов.
            skills: Реестр динамических навыков (если подключен).
            history: Начальная история диалога.
            metadata: Пользовательские метаданные.
        """
        self.session_id = session_id
        self.orchestrator = orchestrator
        self.parent_id = parent_id
        self.role = role
        self.system_prompt = system_prompt
        self.tools: dict[str, Callable[..., Any]] = dict(tools or {})
        self.skills = skills
        self.history: list[dict[str, str]] = list(history or [])
        self.metadata: dict[str, Any] = dict(metadata or {})

    async def execute(
        self,
        prompt: str,
        max_iterations: int = 5,
        preload_skills: list[str] | None = None,
    ) -> str:
        """Запускает цикл автономной работы агента в контексте сессии.

        Args:
            prompt: Входной запрос для агента.
            max_iterations: Максимальное количество итераций tool calling.
            preload_skills: Навыки для немедленной предварительной активации.

        Returns:
            Финальный текстовый ответ агента.
        """
        # Собираем контекст из системного промпта, истории и нового сообщения
        full_prompt = self._build_full_prompt(prompt)

        response = await self.orchestrator.execute_loop(
            prompt=full_prompt,
            tools=self.tools,
            max_iterations=max_iterations,
            skills=self.skills,
            preload_skills=preload_skills,
        )

        # Сохраняем шаг в историю сессии
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": response})

        return response

    def _build_full_prompt(self, user_prompt: str) -> str:
        """Формирует итоговый текст промпта с учетом системных правил и истории."""
        parts: list[str] = []

        if self.system_prompt:
            parts.append(f"SYSTEM INSTRUCTION ({self.role}):\n{self.system_prompt.strip()}")

        if self.history:
            history_lines: list[str] = ["--- DIALOGUE HISTORY ---"]
            for msg in self.history:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                history_lines.append(f"{role}: {content}")
            parts.append("\n".join(history_lines))

        parts.append(f"USER: {user_prompt.strip()}")
        return "\n\n".join(parts)

    def to_snapshot(self) -> SessionSnapshot:
        """Создает снимок состояния текущей сессии для сериализации.

        Returns:
            Экземпляр SessionSnapshot.
        """
        return SessionSnapshot(
            session_id=self.session_id,
            parent_id=self.parent_id,
            role=self.role,
            system_prompt=self.system_prompt,
            history=list(self.history),
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SessionSnapshot,
        orchestrator: AgentOrchestrator,
        tools: dict[str, Callable[..., Any]] | None = None,
        skills: SkillRegistry | None = None,
    ) -> "AgentSession":
        """Восстанавливает агентную сессию из снимка состояния.

        Args:
            snapshot: Снимок сессии.
            orchestrator: Оркестратор.
            tools: Доступные инструменты.
            skills: Реестр навыков.

        Returns:
            Экземпляр AgentSession.
        """
        return cls(
            session_id=snapshot.session_id,
            orchestrator=orchestrator,
            parent_id=snapshot.parent_id,
            role=snapshot.role,
            system_prompt=snapshot.system_prompt,
            tools=tools,
            skills=skills,
            history=snapshot.history,
            metadata=snapshot.metadata,
        )
