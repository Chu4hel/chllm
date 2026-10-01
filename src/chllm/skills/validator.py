"""Валидатор структуры, метаданных и синтаксиса навыков."""

import ast
import re

import yaml

from ..exceptions import SkillValidationError
from .domain import SkillDefinition, SkillMetadata

SKILL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class SkillValidator:
    """Валидатор для проверки корректности навыков."""

    @staticmethod
    def validate_name(name: str) -> None:
        """Проверяет корректность системного имени навыка.

        Args:
            name: Имя навыка.

        Raises:
            SkillValidationError: Если имя пустое или содержит недопустимые символы.
        """
        if not name or not isinstance(name, str):
            raise SkillValidationError("Имя навыка должно быть непустой строкой")

        if not SKILL_NAME_PATTERN.match(name):
            raise SkillValidationError(
                f"Недопустимое имя навыка '{name}'. Разрешены только латинские буквы, цифры, дефис и подчеркивание."
            )

    @staticmethod
    def validate_metadata(metadata: SkillMetadata) -> None:
        """Проверяет полноту метаданных навыка.

        Args:
            metadata: Экземпляр SkillMetadata.

        Raises:
            SkillValidationError: При отсутствии обязательных полей или некорректном имени.
        """
        SkillValidator.validate_name(metadata.name)

        if not metadata.description or not metadata.description.strip():
            raise SkillValidationError(f"Навык '{metadata.name}' должен содержать непустое описание (description)")

        if not metadata.when_to_use or not metadata.when_to_use.strip():
            raise SkillValidationError(f"Навык '{metadata.name}' должен содержать условия активации (when_to_use)")

    @staticmethod
    def validate_definition(definition: SkillDefinition) -> None:
        """Проверяет полное определение навыка.

        Args:
            definition: Экземпляр SkillDefinition.

        Raises:
            SkillValidationError: Если метаданные, инструкция или инструменты невалидны.
        """
        SkillValidator.validate_metadata(definition.metadata)

        if not definition.instruction or not definition.instruction.strip():
            raise SkillValidationError(
                f"Навык '{definition.metadata.name}' должен содержать непустую инструкцию (instruction)"
            )

        for tool_name, tool_func in definition.tools.items():
            if not callable(tool_func):
                msg = f"Инструмент '{tool_name}' в навыке '{definition.metadata.name}' должен быть Callable"
                raise SkillValidationError(msg)

    @staticmethod
    def parse_and_validate_markdown(content: str) -> tuple[SkillMetadata, str]:
        """Парсит markdown-файл навыка с YAML Frontmatter и валидирует его.

        Args:
            content: Текстовое содержимое SKILL.md.

        Returns:
            Кортеж (SkillMetadata, текст_инструкции).

        Raises:
            SkillValidationError: Если Frontmatter отсутствует, поврежден или не содержит нужных полей.
        """
        stripped = content.strip()
        if not stripped.startswith("---"):
            raise SkillValidationError("Файл навыка должен начинаться с YAML Frontmatter (разделитель '---')")

        parts = stripped.split("---", 2)
        if len(parts) < 3:
            raise SkillValidationError("Не найден закрывающий разделитель '---' для YAML Frontmatter в файле навыка")

        frontmatter_str = parts[1].strip()
        instruction_body = parts[2].strip()

        try:
            data = yaml.safe_load(frontmatter_str)
        except Exception as e:
            raise SkillValidationError(f"Ошибка синтаксиса YAML в Frontmatter навыка: {e}") from e

        if not isinstance(data, dict):
            raise SkillValidationError("YAML Frontmatter навыка должен представлять собой словарь ключ-значение")

        try:
            metadata = SkillMetadata(**data)
        except Exception as e:
            raise SkillValidationError(f"Ошибка в полях Frontmatter навыка: {e}") from e

        SkillValidator.validate_metadata(metadata)

        if not instruction_body:
            raise SkillValidationError(f"Навык '{metadata.name}' содержит пустой текст инструкции после Frontmatter")

        return metadata, instruction_body

    @staticmethod
    def validate_python_code(source_code: str, filename: str = "<skill_tools>") -> None:
        """Проверяет синтаксическую корректность Python-кода инструментов.

        Args:
            source_code: Исходный код Python.
            filename: Имя файла для сообщений об ошибках.

        Raises:
            SkillValidationError: При наличии синтаксических ошибок в коде.
        """
        try:
            ast.parse(source_code, filename=filename)
        except SyntaxError as e:
            raise SkillValidationError(f"Синтаксическая ошибка в коде инструментов навыка ({filename}): {e}") from e
