"""Модуль с определениями исключений библиотеки chllm.

Обеспечивает единую иерархию ошибок для различных LLM провайдеров.
"""

from .agentic import AgenticError, SessionNotFoundError, SessionTimeoutError
from .base import CHLLMError
from .parsing import InvalidRequestError, ParsingError
from .provider import (
    AuthenticationError,
    ContentBlockedError,
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
    "ParsingError",
    "QuotaExceededError",
    "RateLimitError",
    "ServiceUnavailableError",
    "SessionNotFoundError",
    "SessionTimeoutError",
    "SkillError",
    "SkillNotFoundError",
    "SkillValidationError",
]
