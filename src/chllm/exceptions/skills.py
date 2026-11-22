"""Исключения для подсистемы динамических навыков (Skills)."""

from .base import CHLLMError


class SkillError(CHLLMError):
    """Базовое исключение для ошибок работы с навыками."""


class SkillValidationError(SkillError):
    """Ошибка валидации структуры, метаданных или файлов навыка."""


class SkillNotFoundError(SkillError):
    """Исключение при попытке доступа к незарегистрированному навыку."""
