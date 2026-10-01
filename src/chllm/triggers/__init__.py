"""Подсистема текстовых fallback-триггеров (chllm.triggers)."""

from .domain import TextCommandTrigger
from .registry import TriggerRegistry

__all__ = [
    "TextCommandTrigger",
    "TriggerRegistry",
]
