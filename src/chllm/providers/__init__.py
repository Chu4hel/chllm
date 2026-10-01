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

__all__ = [
    "GeminiThinkingConfig",
    "GenericCallableProvider",
    "GoogleGenAIProvider",
    "OpenAICompatibleProvider",
    "model_supports_thinking",
]
