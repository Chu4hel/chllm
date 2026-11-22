"""Совместимый с ch_agent модуль команд агентов."""

from chllm.teams import (
    AgentKind,
    AgentSpec,
    AgentTeam,
    build_specialist,
    create_researcher_developer_verifier_team,
)

__all__ = [
    "AgentKind",
    "AgentSpec",
    "AgentTeam",
    "build_specialist",
    "create_researcher_developer_verifier_team",
]
