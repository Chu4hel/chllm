"""Компонент для генерации структурированных промптов для LLM.

Позволяет объединять системные инструкции, контекст, схему ответа,
Few-Shot примеры и входные данные в единый структурированный запрос.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .skills.registry import SkillRegistry
    from .triggers.registry import TriggerRegistry


class FewShotExample(BaseModel):
    """Структура отдельного Few-Shot примера для обучения модели на демонстрациях."""

    input_data: Any = Field(description="Входные данные примера (строка, список, словарь или Pydantic-модель).")
    expected_output: Any = Field(description="Ожидаемый ответ модели (Pydantic-модель, словарь или строка).")
    description: str | None = Field(default=None, description="Опциональное пояснение к примеру.")


class PromptBuilder:
    """Генератор промптов для работы с нейросетями."""

    def __init__(self, template: str | None = None) -> None:
        """Инициализирует генератор промптов.

        Args:
            template: Основная системная инструкция (шаблон).
        """
        self._template = template or ""
        self._examples: list[FewShotExample] = []

    def add_few_shot_example(
        self,
        input_data: Any,
        expected_output: Any,
        description: str | None = None,
    ) -> PromptBuilder:
        """Добавляет один Few-Shot пример демонстрации.

        Args:
            input_data: Входные данные примера.
            expected_output: Ожидаемый результат (Pydantic-модель, словарь или строка).
            description: Опциональное пояснение к примеру.

        Returns:
            Self для цепочечного вызова (Fluent interface).
        """
        self._examples.append(
            FewShotExample(
                input_data=input_data,
                expected_output=expected_output,
                description=description,
            )
        )
        return self

    def add_few_shot_examples(
        self,
        examples: Sequence[tuple[Any, Any] | tuple[Any, Any, str] | FewShotExample | dict[str, Any]],
    ) -> PromptBuilder:
        """Добавляет набор Few-Shot примеров для включения в промпт.

        Args:
            examples: Список кортежей (входные данные, ожидаемый ответ[, описание])
                или объектов FewShotExample.

        Returns:
            Self для цепочечного вызова (Fluent interface).
        """
        for ex in examples:
            if isinstance(ex, FewShotExample):
                self._examples.append(ex)
            elif isinstance(ex, (tuple, list)):
                if len(ex) == 2:
                    self._examples.append(FewShotExample(input_data=ex[0], expected_output=ex[1]))
                elif len(ex) >= 3:
                    self._examples.append(
                        FewShotExample(input_data=ex[0], expected_output=ex[1], description=str(ex[2]))
                    )
            elif isinstance(ex, dict):
                self._examples.append(FewShotExample(**ex))
        return self

    def clear_examples(self) -> PromptBuilder:
        """Очищает список добавленных примеров.

        Returns:
            Self для цепочечного вызова.
        """
        self._examples.clear()
        return self

    @classmethod
    def format_few_shot_examples(
        cls,
        examples: Sequence[tuple[Any, Any] | FewShotExample | dict[str, Any]],
        *,
        label: str = "EXAMPLES",
        input_label: str = "Input",
        output_label: str = "Expected Output",
    ) -> str:
        """Форматирует Few-Shot примеры в стандартный текстовый блок промпта.

        Args:
            examples: Список кортежей или объектов FewShotExample.
            label: Заголовок секции примеров.
            input_label: Заголовок блока входных данных примера.
            output_label: Заголовок блока ожидаемого ответа примера.

        Returns:
            Отформатированная строка с примерами.
        """
        if not examples:
            return ""

        blocks = [f"--- {label} ---"]
        for idx, raw_ex in enumerate(examples, 1):
            if isinstance(raw_ex, FewShotExample):
                ex = raw_ex
            elif isinstance(raw_ex, (tuple, list)):
                desc = str(raw_ex[2]) if len(raw_ex) >= 3 else None
                ex = FewShotExample(input_data=raw_ex[0], expected_output=raw_ex[1], description=desc)
            elif isinstance(raw_ex, dict):
                ex = FewShotExample(**raw_ex)
            else:
                continue

            header = f"Example {idx}:" if not ex.description else f"Example {idx} ({ex.description}):"

            # Сериализуем входные данные
            if isinstance(ex.input_data, (dict, list, BaseModel)):
                input_str = f"```json\n{cls._serialize_data(ex.input_data)}\n```"
            else:
                input_str = str(ex.input_data)

            # Сериализуем ожидаемый ответ
            if isinstance(ex.expected_output, (dict, list, BaseModel)):
                output_str = f"```json\n{cls._serialize_data(ex.expected_output)}\n```"
            else:
                clean_output = str(ex.expected_output).strip()
                output_str = f"```json\n{clean_output}\n```" if not clean_output.startswith("```") else clean_output

            block = f"{header}\n{input_label}:\n{input_str}\n{output_label}:\n{output_str}"
            blocks.append(block)

        return "\n\n".join(blocks)

    def build(
        self,
        input_data: list[Any] | dict[str, Any] | BaseModel | str,
        context: list[str] | None = None,
        instruction: str | list[str] | None = None,
        few_shot_examples: Sequence[tuple[Any, Any] | FewShotExample | dict[str, Any]] | None = None,
        data_label: str = "INPUT DATA",
        context_label: str = "CONTEXT",
        examples_label: str = "EXAMPLES",
        response_model: type[BaseModel] | None = None,
        skills: SkillRegistry | None = None,
        triggers: TriggerRegistry | None = None,
    ) -> str:
        """Собирает финальный текст промпта.

        Args:
            input_data: Данные для обработки (строка, список, словарь или Pydantic-модель).
            context: Список строк контекста (предыдущие события/диалоги).
            instruction: Дополнительная инструкция (строка) или список инструкций.
            few_shot_examples: Опциональный список примеров только для этого вызова.
            data_label: Заголовок для блока данных в промпте.
            context_label: Заголовок для блока контекста в промпте.
            examples_label: Заголовок для блока Few-Shot примеров в промпте.
            response_model: Опциональная Pydantic-модель для автоматической генерации JSON-схемы.
            skills: Опциональный реестр навыков для добавления каталога в промпт.
            triggers: Опциональный реестр текстовых fallback-триггеров для подсказки в промпте.

        Returns:
            Сформированный текст промпта.
        """
        parts = []

        # 1. Сбор инструкций (Системный промпт)
        instructions = []
        if self._template:
            instructions.append(self._template.strip())

        if instruction:
            if isinstance(instruction, list):
                instructions.extend([i.strip() for i in instruction if i])
            else:
                instructions.append(instruction.strip())

        if response_model:
            schema_instruction = self.generate_schema_instruction(response_model)
            instructions.append(schema_instruction)

        if skills is not None and hasattr(skills, "format_catalog"):
            catalog_instruction = skills.format_catalog()
            if catalog_instruction:
                instructions.append(catalog_instruction)

        if triggers is not None and hasattr(triggers, "format_instructions"):
            trigger_instruction = triggers.format_instructions()
            if trigger_instruction:
                instructions.append(trigger_instruction)

        if instructions:
            parts.append("\n".join(instructions))

        # 2. Контекст
        if context:
            parts.append(f"\n--- {context_label} ---")
            parts.append("\n".join(context))

        # 3. Few-Shot примеры
        active_examples: list[Any] = list(self._examples)
        if few_shot_examples:
            active_examples.extend(few_shot_examples)

        if active_examples:
            formatted_examples = self.format_few_shot_examples(active_examples, label=examples_label)
            if formatted_examples:
                parts.append(f"\n{formatted_examples}")

        # 4. Данные (JSON / текст)
        parts.append(f"\n--- {data_label} ---")
        json_data = self._serialize_data(input_data)
        parts.append(json_data)

        return "\n\n".join(parts)

    @staticmethod
    def _serialize_data(data: Any) -> str:
        """Сериализует данные в JSON строку."""
        # 1. Проверяем наличие метода Pydantic v2 (важно для Mock-объектов в тестах)
        if hasattr(data, "model_dump_json") and callable(data.model_dump_json):
            serialized = data.model_dump_json(indent=2)
            if isinstance(serialized, str):
                return serialized
            return str(serialized)

        # 2. Базовая сериализация для стандартных типов (dict, list)
        if isinstance(data, (dict, list)):
            return json.dumps(data, indent=2, ensure_ascii=False)

        # 3. Если это BaseModel (Pydantic v2)
        if isinstance(data, BaseModel):
            return data.model_dump_json(indent=2)

        # 4. Фолбэк на строковое представление
        return str(data)

    @staticmethod
    def generate_schema_instruction(model: type[BaseModel]) -> str:
        """Формирует строгую инструкцию для LLM с JSON-схемой целевой Pydantic-модели.

        Args:
            model: Pydantic-модель для генерации схемы.

        Returns:
            Строка инструкции с JSON-схемой.
        """
        if hasattr(model, "model_json_schema"):
            schema_dict = model.model_json_schema()
        elif hasattr(model, "schema"):
            schema_dict = model.schema()
        else:
            return ""

        schema_json = json.dumps(schema_dict, indent=2, ensure_ascii=False)
        return (
            "ОТВЕТ ДОЛЖЕН БЫТЬ СТРОГО В ФОРМАТЕ JSON, СООТВЕТСТВУЮЩЕМ СЛЕДУЮЩЕЙ СХЕМЕ:\n"
            f"```json\n{schema_json}\n```\n"
            "Не добавляй никаких вводных слов, пояснений или текста вне JSON-блока."
        )
