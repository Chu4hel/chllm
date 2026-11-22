"""Пакет chllm.agent — совместимый с ch_agent интерфейс поверх chllm.

Предоставляет те же имена, что и ``ch_agent``, чтобы упростить миграцию
проектов (включая FWAgentC) на ``chllm``.
"""

from chllm.agentic import (
    AgentMode,
    AgentStepResult,
    AgentTask,
    AgentTaskStatus,
    AgentTaskType,
    IInteractiveQueue,
)
from chllm.buffer import FileTaskQueue, ITaskQueue

try:
    from chllm.agent.teams import (
        AgentSpec,
        AgentTeam,
        build_specialist,
        create_researcher_developer_verifier_team,
    )
    from chllm.integrations.smolagents import (
        BaseSmolToolAdapter,
        BufferSmolModel,
        WebChatSmolModel,
        agent_tool,
        create_agent_model,
    )
    from chllm.integrations.smolagents.models import (
        build_step_limit_instruction,
        compress_history_messages,
    )

    _HAS_SMOLAGENTS = True
except ImportError:
    _HAS_SMOLAGENTS = False
    BaseSmolToolAdapter = None  # type: ignore
    WebChatSmolModel = None  # type: ignore
    BufferSmolModel = None  # type: ignore
    create_agent_model = None  # type: ignore
    agent_tool = None  # type: ignore
    compress_history_messages = None  # type: ignore
    build_step_limit_instruction = None  # type: ignore
    AgentSpec = None  # type: ignore
    AgentTeam = None  # type: ignore
    build_specialist = None  # type: ignore
    create_researcher_developer_verifier_team = None  # type: ignore
from chllm.agentic import InteractiveTaskQueue

__all__ = [
    "AgentMode",
    "AgentStepResult",
    "AgentTask",
    "AgentTaskStatus",
    "AgentTaskType",
    "BaseSmolToolAdapter",
    "BufferSmolModel",
    "FileTaskQueue",
    "IInteractiveQueue",
    "ITaskQueue",
    "InteractiveTaskQueue",
    "WebChatSmolModel",
    "agent_tool",
    "build_step_limit_instruction",
    "compress_history_messages",
    "create_agent_model",
]
