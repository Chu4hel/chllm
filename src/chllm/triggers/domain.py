"""Доменные модели для текстовых fallback-триггеров (chllm.triggers)."""

import re

from pydantic import BaseModel, ConfigDict, Field


class TextCommandTrigger(BaseModel):
    """Определение текстового триггера команды для извлечения ToolCall из текста.

    Attributes:
        name: Системное имя инструмента (ToolCall.name).
        pattern: Скомпилированное регулярное выражение с именованными группами.
        description: Описание назначения команды для системного промпта.
        example: Пример синтаксиса маркера для включения в промпт.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Имя сопоставляемого инструмента")
    pattern: re.Pattern[str] = Field(
        ...,
        description="Регулярное выражение с именованными группами (?P<arg_name>...)",
    )
    description: str = Field(default="", description="Описание назначения команды")
    example: str = Field(default="", description="Пример формата маркера")
