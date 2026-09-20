"""Модуль с определениями исключений библиотеки chllm.

Обеспечивает единую иерархию ошибок для различных LLM провайдеров.
"""

from .base import CHLLMError
from .parsing import InvalidRequestError, ParsingError
from .provider import (
    AuthenticationError,
    ContentBlockedError,
    QuotaExceededError,
    RateLimitError,
    ServiceUnavailableError,
)

__all__ = [
    "AuthenticationError",
    "CHLLMError",
    "ContentBlockedError",
    "InvalidRequestError",
    "ParsingError",
    "QuotaExceededError",
    "RateLimitError",
    "ServiceUnavailableError",
]
