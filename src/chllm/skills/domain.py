"""Доменные модели и протоколы подсистемы динамических навыков (Skills)."""

from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class SkillMetadata(BaseModel):
    """Метаданные навыка для каталога в системном промпте.

    Attributes:
        name: Уникальное системное имя навыка (латиница, цифры, дефис, подчеркивание).
        description: Краткое описание функциональности навыка.
        when_to_use: Четкие триггеры и условия, когда навык должен быть загружен.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Уникальное системное имя навыка")
    description: str = Field(..., description="Краткое описание функциональности навыка")
    when_to_use: str = Field(..., description="Условия и триггеры для активации навыка")


class SkillDefinition(BaseModel):
    """Полное определение навыка с инструкциями, инструментами и ресурсами.

    Attributes:
        metadata: Метаданные навыка.
        instruction: Полный текст системной инструкции или правил поведения.
        tools: Словарь доступных Python-функций, активируемых вместе с навыком.
        resources: Вспомогательные данные (справочники, схемы, словари).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: SkillMetadata = Field(..., description="Метаданные навыка")
    instruction: str = Field(..., description="Текст системной инструкции навыка")
    tools: dict[str, Callable[..., Any]] = Field(
        default_factory=dict,
        description="Исполняемые Python-функции, активируемые с навыком",
    )
    resources: dict[str, Any] = Field(
        default_factory=dict,
        description="Справочные ресурсы навыка",
    )


class ISkillSource(Protocol):
    """Протокол источника навыков (Dependency Inversion)."""

    async def get_catalog(self) -> list[SkillMetadata]:
        """Возвращает метаданные всех доступных навыков из источника.

        Returns:
            Список метаданных навыков для включения в промпт.
        """
        ...

    async def load_skill(self, name: str) -> SkillDefinition | None:
        """Загружает полное определение навыка по его имени.

        Args:
            name: Имя запрашиваемого навыка.

        Returns:
            Экземпляр SkillDefinition или None, если навык не найден в источнике.
        """
        ...
