"""Исключения при валидации запросов и разборе ответов LLM."""

from .base import CHLLMError


class InvalidRequestError(CHLLMError):
    """Ошибка в структуре запроса или параметрах."""


class ParsingError(CHLLMError):
    """Ошибка при разборе ответа от LLM."""
