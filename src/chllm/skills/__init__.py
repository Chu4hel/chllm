"""Подсистема динамических навыков (Lazy Skills System) для chllm."""

from .domain import ISkillSource, SkillDefinition, SkillMetadata
from .registry import SkillRegistry
from .sources import FileSystemSkillSource, InMemorySkillSource
from .validator import SkillValidator

__all__ = [
    "FileSystemSkillSource",
    "ISkillSource",
    "InMemorySkillSource",
    "SkillDefinition",
    "SkillMetadata",
    "SkillRegistry",
    "SkillValidator",
]
