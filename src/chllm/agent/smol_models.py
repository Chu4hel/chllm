"""Совместимый с ch_agent модуль моделей smolagents."""

from chllm.integrations.smolagents.models import (
    BufferSmolModel,
    WebChatSmolModel,
    build_step_limit_instruction,
    compress_history_messages,
    create_agent_model,
)

__all__ = [
    "BufferSmolModel",
    "WebChatSmolModel",
    "build_step_limit_instruction",
    "compress_history_messages",
    "create_agent_model",
]
