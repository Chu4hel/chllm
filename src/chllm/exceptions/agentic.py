"""Исключения для мультиагентной подсистемы chllm."""

from .base import CHLLMError


class AgenticError(CHLLMError):
    """Базовое исключение для ошибок мультиагентного взаимодействия."""


class SessionNotFoundError(AgenticError):
    """Исключение, если указанная сессия не найдена в пуле или хранилище."""


class SessionTimeoutError(AgenticError):
    """Исключение при превышении таймаута ожидания ответа от сессии."""
