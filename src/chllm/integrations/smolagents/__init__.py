"""Интеграция с библиотекой smolagents от Hugging Face."""

from typing import Any

try:
    import smolagents

    _HAS_SMOLAGENTS = True
except ImportError:
    _HAS_SMOLAGENTS = False

from .tools import BaseSmolToolAdapter

if _HAS_SMOLAGENTS:
    from .models import (
        BufferSmolModel,
        WebChatSmolModel,
        create_agent_model,
    )

    __all__ = [
        "BaseSmolToolAdapter",
        "BufferSmolModel",
        "WebChatSmolModel",
        "create_agent_model",
    ]
else:
    __all__ = [
        "BaseSmolToolAdapter",
        "BufferSmolModel",
        "WebChatSmolModel",
        "create_agent_model",
    ]

    def __getattr__(name: str) -> Any:
        if name in ("BufferSmolModel", "WebChatSmolModel", "create_agent_model"):
            raise ImportError(
                f"Компонент '{name}' требует установленной библиотеки smolagents. "
                "Установите ее: pip install 'chllm[smolagents]'"
            )
        raise AttributeError(f"Модуль '{__name__}' не содержит атрибут '{name}'")
