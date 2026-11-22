"""Совместимый с ch_agent модуль интерфейсов."""

from chllm.agentic import IInteractiveQueue
from chllm.buffer import ITaskQueue

__all__ = ["IInteractiveQueue", "ITaskQueue"]
