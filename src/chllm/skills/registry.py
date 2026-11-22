"""Реестр динамических навыков (SkillRegistry)."""

import asyncio

from .domain import ISkillSource, SkillDefinition, SkillMetadata
from .sources import InMemorySkillSource


class SkillRegistry:
    """Центральный реестр навыков для оркестрации и генерации промптов."""

    def __init__(self, sources: list[ISkillSource] | None = None) -> None:
        """Инициализирует реестр навыков.

        Args:
            sources: Список начальных источников навыков.
        """
        self._sources: list[ISkillSource] = list(sources or [])

    def register_source(self, source: ISkillSource) -> None:
        """Регистрирует новый источник навыков в реестре.

        Args:
            source: Источник навыков, реализующий ISkillSource.
        """
        self._sources.append(source)

    def register_skill(self, skill: SkillDefinition) -> None:
        """Удобный метод для прямой регистрации скилла в памяти.

        Args:
            skill: Экземпляр SkillDefinition.
        """
        mem_source: InMemorySkillSource | None = None
        for src in self._sources:
            if isinstance(src, InMemorySkillSource):
                mem_source = src
                break

        if mem_source is None:
            mem_source = InMemorySkillSource()
            self._sources.append(mem_source)

        mem_source.add(skill)

    async def get_catalog(self) -> list[SkillMetadata]:
        """Возвращает объединенный каталог всех доступных навыков.

        Returns:
            Список уникальных метаданных навыков.
        """
        catalog: dict[str, SkillMetadata] = {}
        for source in self._sources:
            items = await source.get_catalog()
            for item in items:
                if item.name not in catalog:
                    catalog[item.name] = item
        return list(catalog.values())

    async def load_skill(self, name: str) -> SkillDefinition | None:
        """Загружает полное определение навыка по имени из зарегистрированных источников.

        Args:
            name: Системное имя навыка.

        Returns:
            Экземпляр SkillDefinition или None, если навык не найден.
        """
        for source in self._sources:
            skill = await source.load_skill(name)
            if skill is not None:
                return skill
        return None

    def format_catalog(self, catalog: list[SkillMetadata] | None = None) -> str:
        """Формирует компактный текстовый каталог для включения в промпт.

        Args:
            catalog: Опциональный уже полученный список метаданных. Если не передан,
                каталог считывается синхронно из тех источников, которые поддерживают это,
                либо формируется из переданного списка.

        Returns:
            Форматированная строка со списком доступных навыков.
        """
        if catalog is None:
            # Пытаемся получить синхронно, если доступно
            catalog = self._get_catalog_sync_fallback()

        if not catalog:
            return ""

        lines = [
            "--- AVAILABLE SKILLS ---",
            "You have access to specialized skills that can be loaded on demand.",
        ]
        for item in catalog:
            lines.append(f"- {item.name}: {item.description} (When to use: {item.when_to_use})")

        lines.append("To activate a skill, call tool `load_skill(name='...')` or output marker `[LOAD_SKILL: <name>]`.")
        return "\n".join(lines)

    def _get_catalog_sync_fallback(self) -> list[SkillMetadata]:
        """Синхронный сборщик каталога для источников с синхронным кэшем."""
        catalog: dict[str, SkillMetadata] = {}
        for source in self._sources:
            # Если это InMemorySkillSource или FileSystemSkillSource с готовым кэшем
            if hasattr(source, "_catalog_cache") and isinstance(source._catalog_cache, dict):
                for name, meta in source._catalog_cache.items():
                    catalog.setdefault(name, meta)
            elif hasattr(source, "_skills") and isinstance(source._skills, dict):
                for name, skill in source._skills.items():
                    catalog.setdefault(name, skill.metadata)
            else:
                # Если асинхронный произвольный источник
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # Уже в цикле событий, пропускаем синхронный опрос
                    pass
                else:
                    items = asyncio.run(source.get_catalog())
                    for item in items:
                        catalog.setdefault(item.name, item)
        return list(catalog.values())
