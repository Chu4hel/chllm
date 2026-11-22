"""Совместимый с ch_agent модуль инструментов."""

from chllm.integrations.smolagents.tools import (
    BaseSmolToolAdapter,
    agent_tool,
)

__all__ = ["BaseSmolToolAdapter", "agent_tool"]
