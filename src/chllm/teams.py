"""Multi-Agent Teams: шаблоны координатора и специализированных агентов smolagents.

Модуль предоставляет удобные фабрики для сборки команды агентов
(Исследователь, Разработчик, Верификатор и т.д.) на базе
``smolagents.CodeAgent`` / ``smolagents.ToolCallingAgent`` с общим
координатором, управляющим специализированными агентами через
механизм ``managed_agents``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from chutils import setup_logger

logger = setup_logger("ch_agent.teams")

AgentKind = Literal["code", "tool_calling"]


@dataclass
class AgentSpec:
    """Спецификация специализированного агента команды.

    Args:
        name: Уникальное имя агента (используется координатором).
        description: Описание роли агента для координатора.
        tools: Список инструментов smolagents.
        system_prompt: Опциональный системный промпт.
        max_steps: Лимит шагов агента.
        kind: Тип агента (``code`` или ``tool_calling``).
    """

    name: str
    description: str
    tools: list[Any] = field(default_factory=list)
    system_prompt: str | None = None
    max_steps: int = 10
    kind: AgentKind = "code"


def build_specialist(
    spec: AgentSpec,
    model: Any,
    *,
    agent_factory: Callable[..., Any] | None = None,
) -> Any:
    """Создаёт специализированного агента по спецификации.

    Args:
        spec: Спецификация агента.
        model: Модель smolagents (Model).
        agent_factory: Опциональная фабрика агента (по умолчанию выбирается
            по ``spec.kind``).

    Returns:
        Экземпляр ``CodeAgent`` или ``ToolCallingAgent`` с именем и описанием.
    """
    from smolagents import CodeAgent, ToolCallingAgent

    if agent_factory is not None:
        agent = agent_factory(
            tools=spec.tools,
            model=model,
            max_steps=spec.max_steps,
        )
    elif spec.kind == "tool_calling":
        agent = ToolCallingAgent(
            tools=spec.tools,
            model=model,
            max_steps=spec.max_steps,
        )
    else:
        agent = CodeAgent(
            tools=spec.tools,
            model=model,
            max_steps=spec.max_steps,
        )

    agent.name = spec.name
    agent.description = spec.description
    if spec.system_prompt:
        # smolagents хранит системный промпт в prompt_templates; задаём через атрибут,
        # если поддерживается, иначе — игнорируем без ошибки.
        try:
            agent.prompt_templates = getattr(agent, "prompt_templates", {})
            if isinstance(agent.prompt_templates, dict):
                agent.prompt_templates["system_prompt"] = spec.system_prompt
        except Exception:
            logger.debug("Не удалось установить system_prompt для %s", spec.name)
    return agent


class AgentTeam:
    """Команда агентов с координатором и специализированными агентами.

    Координатор (обычно ``CodeAgent``) получает список специалистов через
    ``managed_agents`` и может делегировать им подзадачи.

    Args:
        coordinator_model: Модель для координатора.
        specialists: Список спецификаций специалистов.
        coordinator_tools: Инструменты, доступные координатору.
        coordinator_max_steps: Лимит шагов координатора.
        coordinator_kind: Тип координатора.
    """

    def __init__(
        self,
        coordinator_model: Any,
        specialists: list[AgentSpec],
        *,
        coordinator_tools: list[Any] | None = None,
        coordinator_max_steps: int = 20,
        coordinator_kind: AgentKind = "code",
        specialist_model: Any | None = None,
    ) -> None:
        self.coordinator_model = coordinator_model
        self.specialist_model = specialist_model or coordinator_model
        self.specialists_specs = specialists
        self.coordinator_tools = coordinator_tools or []
        self.coordinator_max_steps = coordinator_max_steps
        self.coordinator_kind = coordinator_kind

        self.specialists: list[Any] = [build_specialist(spec, self.specialist_model) for spec in specialists]
        self.coordinator = self._build_coordinator()

    def _build_coordinator(self) -> Any:
        from smolagents import CodeAgent, ToolCallingAgent

        if self.coordinator_kind == "tool_calling":
            coord = ToolCallingAgent(
                tools=self.coordinator_tools,
                model=self.coordinator_model,
                managed_agents=self.specialists,
                max_steps=self.coordinator_max_steps,
            )
        else:
            coord = CodeAgent(
                tools=self.coordinator_tools,
                model=self.coordinator_model,
                managed_agents=self.specialists,
                max_steps=self.coordinator_max_steps,
            )
        coord.name = "coordinator"
        coord.description = "Координатор команды специализированных агентов."
        return coord

    def run(self, task: str, **kwargs: Any) -> Any:
        """Запускает команду на задаче.

        Args:
            task: Текст задачи для координатора.
            **kwargs: Дополнительные аргументы ``agent.run``.

        Returns:
            Результат работы координатора.
        """
        return self.coordinator.run(task, **kwargs)

    async def arun(self, task: str, **kwargs: Any) -> Any:
        """Асинхронная обёртка над ``run`` (выполняется в отдельном потоке)."""
        import asyncio

        return await asyncio.to_thread(self.coordinator.run, task, **kwargs)


def create_researcher_developer_verifier_team(
    model: Any,
    *,
    researcher_tools: list[Any] | None = None,
    developer_tools: list[Any] | None = None,
    verifier_tools: list[Any] | None = None,
    coordinator_tools: list[Any] | None = None,
) -> AgentTeam:
    """Создаёт шаблон команды: Исследователь / Разработчик / Верификатор.

    Args:
        model: Модель для всех агентов.
        researcher_tools: Инструменты исследователя.
        developer_tools: Инструменты разработчика.
        verifier_tools: Инструменты верификатора.
        coordinator_tools: Инструменты координатора.

    Returns:
        Готовая ``AgentTeam``.
    """
    specs = [
        AgentSpec(
            name="researcher",
            description="Исследует задачу, собирает контекст и факты.",
            tools=researcher_tools or [],
            kind="code",
        ),
        AgentSpec(
            name="developer",
            description="Реализует изменения кода по результатам исследования.",
            tools=developer_tools or [],
            kind="code",
        ),
        AgentSpec(
            name="verifier",
            description="Проверяет изменения тестами и отчитывается о результатах.",
            tools=verifier_tools or [],
            kind="code",
        ),
    ]
    return AgentTeam(
        coordinator_model=model,
        specialists=specs,
        coordinator_tools=coordinator_tools or [],
    )
