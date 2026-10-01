"""Провайдеры-адаптеры для работы с LLM.

Обеспечивают интеграцию с произвольными асинхронными функциями,
OpenAI-совместимыми сервисами и Google GenAI SDK.
"""

from .callable import GenericCallableProvider
from .google_genai import (
    GeminiThinkingConfig,
    GoogleGenAIProvider,
    model_supports_thinking,
)
from .openai import OpenAICompatibleProvider
from .pool import KeyItem, KeyPoolProvider, KeyStatus, RotationStrategy
from .router import OpenAIRouterProvider

__all__ = [
    "GeminiThinkingConfig",
    "GenericCallableProvider",
    "GoogleGenAIProvider",
    "KeyItem",
    "KeyPoolProvider",
    "KeyStatus",
    "OpenAICompatibleProvider",
    "OpenAIRouterProvider",
    "RotationStrategy",
    "model_supports_thinking",
]
