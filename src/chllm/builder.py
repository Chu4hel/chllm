"""
Компонент для генерации структурированных промптов для LLM.
Позволяет объединять инструкции, контекст и данные в единый запрос.
"""

import json
from typing import Any

from pydantic import BaseModel


class PromptBuilder:
    """Генератор промптов для работы с нейросетями."""

    def __init__(self, template: str | None = None):
        """
        Args:
            template: Основная системная инструкция (шаблон).
        """
        self._template = template or ""

    def build(
        self,
        input_data: list[Any] | dict[str, Any] | BaseModel,
        context: list[str] | None = None,
        instruction: str | list[str] | None = None,
        data_label: str = "INPUT DATA",
        context_label: str = "CONTEXT",
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Собирает финальный текст промпта.

        Args:
            input_data: Данные для обработки (список, словарь или Pydantic-модель).
            context: Список строк контекста (предыдущие события/диалоги).
            instruction: Дополнительная инструкция (строка) или список инструкций.
            data_label: Заголовок для блока данных в промпте.
            context_label: Заголовок для блока контекста в промпте.
            response_model: Опциональная Pydantic-модель для автоматической генерации JSON-схемы.

        Returns:
            str: Сформированный текст промпта.
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

        if instructions:
            parts.append("\n".join(instructions))

        # 2. Контекст
        if context:
            parts.append(f"\n--- {context_label} ---")
            parts.append("\n".join(context))

        # 3. Данные (JSON)
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
            # Если это Mock, который вернул другой Mock, пробуем преобразовать в строку
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
