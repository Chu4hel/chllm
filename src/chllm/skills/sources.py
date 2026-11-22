"""Источники навыков (FileSystem и InMemory)."""

import importlib.util
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from ..exceptions import SkillValidationError
from .domain import ISkillSource, SkillDefinition, SkillMetadata
from .validator import SkillValidator


class InMemorySkillSource(ISkillSource):
    """Источник навыков, хранящихся в оперативной памяти."""

    def __init__(self, skills: list[SkillDefinition] | None = None) -> None:
        """Инициализирует источник навыков в памяти.

        Args:
            skills: Начальный список определений навыков.
        """
        self._skills: dict[str, SkillDefinition] = {}
        if skills:
            for skill in skills:
                self.add(skill)

    def add(self, skill: SkillDefinition) -> None:
        """Регистрирует новый навык в источнике.

        Args:
            skill: Экземпляр SkillDefinition.

        Raises:
            SkillValidationError: Если определение навыка не прошло валидацию.
        """
        SkillValidator.validate_definition(skill)
        self._skills[skill.metadata.name] = skill

    async def get_catalog(self) -> list[SkillMetadata]:
        """Возвращает метаданные всех доступных навыков.

        Returns:
            Список метаданных навыков.
        """
        return [skill.metadata for skill in self._skills.values()]

    async def load_skill(self, name: str) -> SkillDefinition | None:
        """Загружает определение навыка по имени.

        Args:
            name: Имя навыка.

        Returns:
            Определение навыка или None.
        """
        return self._skills.get(name)


class FileSystemSkillSource(ISkillSource):
    """Источник навыков из файловой системы.

    Поддерживает структуру папок:
        skills_dir/
            my_skill/
                SKILL.md (или skill.md)
                tools.py (опционально)
                resources/ (опционально)
            another_skill.md (плоский файл)
    """

    def __init__(self, path: str | Path, validate_on_init: bool = True) -> None:
        """Инициализирует файловый источник навыков.

        Args:
            path: Путь к корневой директории со скиллами.
            validate_on_init: Выполнять ли немедленную Fail-Fast валидацию всех навыков.

        Raises:
            SkillValidationError: При обнаружении ошибок структуры или синтаксиса.
        """
        self._root_path = Path(path)
        self._catalog_cache: dict[str, SkillMetadata] = {}
        self._skills_cache: dict[str, SkillDefinition] = {}
        self._locations: dict[str, Path] = {}

        if not self._root_path.exists():
            raise SkillValidationError(f"Директория навыков не существует: {self._root_path}")
        if not self._root_path.is_dir():
            raise SkillValidationError(f"Указанный путь навыков не является директорией: {self._root_path}")

        self._discover_and_index()

        if validate_on_init:
            self._validate_all()

    def _discover_and_index(self) -> None:
        """Сканирует директорию и индексирует найденные навыки."""
        self._catalog_cache.clear()
        self._locations.clear()

        # 1. Поиск вложенных папок с SKILL.md
        for item in self._root_path.iterdir():
            if item.is_dir():
                skill_file = self._find_skill_file_in_dir(item)
                if skill_file:
                    metadata, _ = self._read_and_parse_file(skill_file)
                    self._catalog_cache[metadata.name] = metadata
                    self._locations[metadata.name] = item

            # 2. Поиск плоских .md файлов
            elif item.is_file() and item.suffix.lower() == ".md":
                metadata, _ = self._read_and_parse_file(item)
                self._catalog_cache[metadata.name] = metadata
                self._locations[metadata.name] = item

    @staticmethod
    def _find_skill_file_in_dir(directory: Path) -> Path | None:
        """Ищет файл инструкции скилла в директории (SKILL.md или skill.md)."""
        for name in ("SKILL.md", "skill.md", f"{directory.name}.md"):
            candidate = directory / name
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _read_and_parse_file(self, file_path: Path) -> tuple[SkillMetadata, str]:
        """Считывает и парсит файл навыка."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise SkillValidationError(f"Не удалось прочитать файл навыка {file_path}: {e}") from e

        return SkillValidator.parse_and_validate_markdown(content)

    def _validate_all(self) -> None:
        """Выполняет полную проверку синтаксиса и файлов для всех проиндексированных навыков."""
        for skill_name in list(self._catalog_cache.keys()):
            self._load_skill_sync(skill_name)

    def _load_skill_sync(self, name: str) -> SkillDefinition | None:
        """Синхронно загружает и кэширует скилл."""
        if name in self._skills_cache:
            return self._skills_cache[name]

        location = self._locations.get(name)
        if not location:
            return None

        if location.is_file():
            metadata, instruction = self._read_and_parse_file(location)
            definition = SkillDefinition(
                metadata=metadata,
                instruction=instruction,
                tools={},
                resources={},
            )
            self._skills_cache[name] = definition
            return definition

        # Если location - это директория
        skill_file = self._find_skill_file_in_dir(location)
        if not skill_file:
            raise SkillValidationError(f"В директории навыка {location} не найден SKILL.md")

        metadata, instruction = self._read_and_parse_file(skill_file)

        # Загрузка tools.py
        tools_dict = self._load_tools_from_dir(location)

        # Загрузка resources
        resources_dict = self._load_resources_from_dir(location)

        definition = SkillDefinition(
            metadata=metadata,
            instruction=instruction,
            tools=tools_dict,
            resources=resources_dict,
        )
        SkillValidator.validate_definition(definition)
        self._skills_cache[name] = definition
        return definition

    def _load_tools_from_dir(self, directory: Path) -> dict[str, Callable[..., Any]]:
        """Загружает функции из tools.py, если файл существует."""
        tools_file = directory / "tools.py"
        if not tools_file.exists():
            return {}

        code = tools_file.read_text(encoding="utf-8")
        SkillValidator.validate_python_code(code, filename=str(tools_file))

        module_name = f"chllm_skill_tools_{directory.name}"
        spec = importlib.util.spec_from_file_location(module_name, tools_file)
        if spec is None or spec.loader is None:
            raise SkillValidationError(f"Не удалось инициализировать загрузчик модуля для {tools_file}")

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise SkillValidationError(f"Ошибка при импорте {tools_file}: {e}") from e

        # Проверяем явный экспорт через словарь TOOLS / tools
        if hasattr(module, "TOOLS") and isinstance(getattr(module, "TOOLS"), dict):
            return getattr(module, "TOOLS")
        if hasattr(module, "tools") and isinstance(getattr(module, "tools"), dict):
            return getattr(module, "tools")
        if hasattr(module, "get_tools") and callable(getattr(module, "get_tools")):
            loaded = getattr(module, "get_tools")()
            if isinstance(loaded, dict):
                return loaded

        # Иначе собираем все публичные функции модуля
        extracted: dict[str, Callable[..., Any]] = {}
        for attr_name, attr_val in module.__dict__.items():
            if (
                not attr_name.startswith("_")
                and inspect.isroutine(attr_val)
                and getattr(attr_val, "__module__", None) == module_name
            ):
                extracted[attr_name] = attr_val
        return extracted

    def _load_resources_from_dir(self, directory: Path) -> dict[str, Any]:
        """Считывает вспомогательные ресурсы из папки resources/."""
        res_dir = directory / "resources"
        if not res_dir.exists() or not res_dir.is_dir():
            return {}

        resources: dict[str, Any] = {}
        for res_file in res_dir.iterdir():
            if not res_file.is_file():
                continue
            name = res_file.name
            suffix = res_file.suffix.lower()
            try:
                content = res_file.read_text(encoding="utf-8")
                if suffix == ".json":
                    resources[name] = json.loads(content)
                elif suffix in (".yaml", ".yml"):
                    resources[name] = yaml.safe_load(content)
                else:
                    resources[name] = content
            except Exception as e:
                raise SkillValidationError(f"Ошибка при чтении ресурса {res_file}: {e}") from e
        return resources

    async def get_catalog(self) -> list[SkillMetadata]:
        """Возвращает метаданные всех навыков из директории.

        Returns:
            Список метаданных навыков.
        """
        return list(self._catalog_cache.values())

    async def load_skill(self, name: str) -> SkillDefinition | None:
        """Асинхронно загружает определение навыка по имени.

        Args:
            name: Имя навыка.

        Returns:
            Определение навыка или None, если он не найден.
        """
        return self._load_skill_sync(name)
