"""Модуль с определениями исключений библиотеки chllm.

Обеспечивает единую иерархию ошибок для различных LLM провайдеров.
"""

from .agentic import AgenticError, SessionNotFoundError, SessionTimeoutError
from .base import CHLLMError
from .parsing import InvalidRequestError, ParsingError
from .pool import NoActiveKeysError
from .provider import (
    AuthenticationError,
    ContentBlockedError,
    LocationUnsupportedError,
    ProxyConnectionError,
    QuotaExceededError,
    RateLimitError,
    ServiceUnavailableError,
)
from .skills import SkillError, SkillNotFoundError, SkillValidationError
from .tools import CliNotFoundError, CliProcessError, CliTimeoutError, CliToolError

__all__ = [
    "AgenticError",
    "AuthenticationError",
    "CHLLMError",
    "CliNotFoundError",
    "CliProcessError",
    "CliTimeoutError",
    "CliToolError",
    "ContentBlockedError",
    "InvalidRequestError",
    "LocationUnsupportedError",
    "NoActiveKeysError",
    "ParsingError",
    "ProxyConnectionError",
    "QuotaExceededError",
    "RateLimitError",
    "ServiceUnavailableError",
    "SessionNotFoundError",
    "SessionTimeoutError",
    "SkillError",
    "SkillNotFoundError",
    "SkillValidationError",
]
