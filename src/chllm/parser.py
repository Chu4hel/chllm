"""
Компонент для робастного парсинга ответов от LLM.
Централизует логику очистки, исправления и разбора JSON данных.
"""

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

try:
    from chutils.logger import setup_logger
except ImportError:
    import logging  # chutils: ignore[ChutilsIntegrationRule]

    def setup_logger(name: str = "chllm") -> Any:
        """Создает резервный логгер, если chutils не установлен.

        Args:
            name: Имя логгера.

        Returns:
            Экземпляр стандартного логгера.
        """
        return logging.getLogger(name)


logger = setup_logger(__name__)

T = TypeVar("T")


class ToolCall(BaseModel):
    """Модель вызова инструмента ИИ."""

    name: str
    args: dict[str, Any] = {}


class RobustLLMParser:
    """Универсальный парсер для обработки ответов от нейросетей."""

    def __init__(self, logger: Any | None = None):
        self._logger = logger or setup_logger(__name__)

    def parse(
        self,
        raw_text: str,
        validation_model: type[T] | None = None,
        glossary_model: type[Any] | None = None,
    ) -> dict[str, list[T | dict[str, Any]]]:
        """Парсит сырой текст ответа ИИ в структурированный словарь.

        Args:
            raw_text: Текст ответа от ИИ.
            validation_model: Опциональная Pydantic-модель для элементов батча.
            glossary_model: Опциональная Pydantic-модель для терминов глоссария.

        Returns:
            Dict[str, List]: Словарь с ключами 'batch' и 'suggested_glossary_terms'.
        """
        if not raw_text or not raw_text.strip():
            return {"batch": [], "suggested_glossary_terms": []}

        # 1. Извлекаем все сбалансированные JSON блоки из текста
        blocks = self._find_json_blocks(raw_text)

        validated_items: list[T | dict[str, Any]] = []
        suggested_glossary_terms: list[Any] = []

        if not blocks:
            # Fallback на старую логику, если блоки не найдены (редкий случай)
            clean_text = self._clean_markdown(raw_text)
            blocks = [clean_text]

        # Очередь блоков для обработки (позволяет рекурсивно заглядывать внутрь поврежденных контейнеров)
        blocks_queue = list(blocks)

        while blocks_queue:
            block = blocks_queue.pop(0)
            if not block:
                continue

            try:
                data = json.loads(block)
                self._process_json_data(
                    data,
                    validated_items,
                    suggested_glossary_terms,
                    validation_model,
                    glossary_model,
                )
            except (json.JSONDecodeError, TypeError):
                # Если блок начинается на '[' и не парсится — возможно, внутри есть мусор.
                # Пробуем найти блоки внутри него и добавить в очередь.
                stripped_block = block.strip()
                if stripped_block.startswith("[") and stripped_block.endswith("]"):
                    inner_content = stripped_block[1:-1]
                    inner_blocks = self._find_json_blocks(inner_content)
                    if inner_blocks:
                        blocks_queue.extend(inner_blocks)
                        continue

                # Попытка восстановления для каждого блока в отдельности
                recovered = self._attempt_json_recovery(block)
                try:
                    data = json.loads(recovered)
                    items_before = len(validated_items)
                    terms_before = len(suggested_glossary_terms)
                    self._process_json_data(
                        data,
                        validated_items,
                        suggested_glossary_terms,
                        validation_model,
                        glossary_model,
                    )

                    if len(validated_items) > items_before or len(suggested_glossary_terms) > terms_before:
                        self._logger.info(
                            "Успешно исправлен и распарсен усеченный JSON: '%s...'",
                            block[:50],
                        )
                except json.JSONDecodeError:
                    continue
        return {
            "batch": validated_items,
            "suggested_glossary_terms": suggested_glossary_terms,
        }

    def _find_json_blocks(self, text: str) -> list[str]:
        """Находит все сбалансированные JSON-блоки ({...} или [...]) в тексте."""
        if not text:
            return []

        blocks = []
        n = len(text)
        i = 0
        while i < n:
            if text[i] in ("{", "["):
                start = i
                opener = text[i]
                closer = "}" if opener == "{" else "]"
                depth = 1
                in_string = False
                escaped = False
                j = i + 1
                while j < n and depth > 0:
                    char = text[j]
                    if char == '"' and not escaped:
                        in_string = not in_string
                    elif in_string:
                        escaped = bool(char == "\\" and not escaped)
                    else:
                        if char == opener:
                            depth += 1
                        elif char == closer:
                            depth -= 1
                    j += 1

                if depth == 0:
                    blocks.append(text[start:j])
                    i = j  # Пропускаем весь блок
                    continue
                else:
                    # Если блок не закрыт, возможно внутри есть валидные блоки?
                    # Продолжаем поиск со следующего символа, не считая этот за начало.
                    pass
            i += 1
        return blocks

    def parse_raw(self, raw_text: str) -> object:
        """Просто парсит JSON с восстановлением, возвращая чистый Python объект.

        Универсальный метод для любых структур данных.

        Args:
            raw_text: Сырой текст ответа ИИ с JSON.

        Returns:
            Распарсенный Python объект (словарь, список или примитив).
        """
        clean = self._clean_markdown(raw_text)
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            recovered = self._attempt_json_recovery(clean)
            return json.loads(recovered)

    @staticmethod
    def parse_tool_calls(text: str) -> list[ToolCall]:
        """
        Ищет и парсит вызовы инструментов (JSON блоки) в тексте.

        Args:
            text: Сырой текст ответа ИИ.

        Returns:
            Список объектов ToolCall.
        """
        if not text:
            return []

        # Находим все сбалансированные JSON-блоки в тексте
        blocks = []
        "Список найденных сбалансированных JSON-строк"
        n = len(text)
        "Длина входного текста"
        i = 0
        "Текущий индекс в тексте"
        while i < n:
            if text[i] == "{":
                start = i
                "Индекс открывающей скобки"
                depth = 1
                "Текущая глубина вложенности скобок"
                in_string = False
                "Флаг нахождения внутри строкового литерала"
                escaped = False
                "Флаг экранирования символа"
                i += 1
                while i < n and depth > 0:
                    char = text[i]
                    "Текущий символ"
                    if char == '"' and not escaped:
                        in_string = not in_string
                    elif in_string:
                        escaped = bool(char == "\\" and not escaped)
                    else:
                        if char == "{":
                            depth += 1
                        elif char == "}":
                            depth -= 1
                    i += 1
                if depth == 0:
                    blocks.append(text[start:i])
            else:
                i += 1

        results = []
        "Список успешно распарсенных вызовов инструментов"
        for block in blocks:
            try:
                data = json.loads(block)
                "Распарсенный JSON-блок в виде словаря"
                if not isinstance(data, dict):
                    continue

                # Поддержка оберток: "tool_use", "tool", "call", "function"
                for wrapper in ("tool_use", "tool", "call", "function"):
                    if wrapper in data and isinstance(data[wrapper], dict):
                        inner = data[wrapper]
                        "Внутренний словарь из обертки"
                        if "name" in inner:
                            data = inner
                            break

                if "name" not in data or not isinstance(data["name"], str):
                    continue

                name = data["name"]
                "Имя вызываемого инструмента"
                args = {}
                "Словарь аргументов инструмента"

                # Ищем аргументы в стандартных полях
                for arg_field in ("args", "parameters", "arguments", "params"):
                    if arg_field in data and isinstance(data[arg_field], dict):
                        args = data[arg_field]
                        break
                else:
                    # Плоские аргументы (все ключи кроме имени и служебных)
                    excluded_keys = {"name", "tool_use", "tool", "call", "function"}
                    "Ключи, которые не интерпретируются как аргументы"
                    args = {k: v for k, v in data.items() if k not in excluded_keys}

                results.append(ToolCall(name=name, args=args))
            except (json.JSONDecodeError, ValueError):
                # Если это не JSON или не валидный ToolCall - игнорируем
                continue

        return results

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Извлекает чистое содержимое JSON из текста (игнорируя болтовню нейросети)."""
        if not text:
            return ""

        # 1. Если есть блок кода markdown - извлекаем ТОЛЬКО его содержимое
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()

        # 2. Если блоков маркдауна нет, жестко обрезаем текст по границам JSON
        # Ищем первую { или [ и последнюю } или ]
        start_obj = text.find("{")
        start_arr = text.find("[")

        starts = [i for i in (start_obj, start_arr) if i != -1]
        if not starts:
            return text.strip()
        start_idx = min(starts)

        end_obj = text.rfind("}")
        end_arr = text.rfind("]")

        ends = [i for i in (end_obj, end_arr) if i != -1]
        if not ends:
            return text.strip()
        end_idx = max(ends)

        if end_idx >= start_idx:
            return text[start_idx : end_idx + 1]

        return text.strip()

    def _process_json_data(
        self,
        data: object,
        items: list[Any],
        terms: list[Any],
        val_model: type[T] | None = None,
        glos_model: type[object] | None = None,
    ) -> None:
        """Разбирает распарсенный JSON и наполняет коллекции.

        Args:
            data: Распарсенные данные (список или словарь).
            items: Коллекция основных элементов для наполнения.
            terms: Коллекция терминов глоссария.
            val_model: Класс модели для валидации основных элементов.
            glos_model: Класс модели для валидации терминов глоссария.
        """
        if isinstance(data, list):
            for entry in data:
                self._process_single_entry(entry, items, terms, val_model, glos_model)
        elif isinstance(data, dict):
            # 1. Проверяем наличие известных контейнеров проекта перевода
            has_batch = "batch" in data and isinstance(data["batch"], list)
            has_glossary = "suggested_glossary_terms" in data and isinstance(data["suggested_glossary_terms"], list)

            if has_batch:
                for entry in data["batch"]:
                    self._process_single_entry(entry, items, terms, val_model, glos_model)

            if has_glossary:
                for term_data in data["suggested_glossary_terms"]:
                    if isinstance(term_data, dict) and "term" in term_data:
                        self._add_item(term_data, terms, glos_model)

            # 2. Если это одиночный объект (не контейнер)
            if not has_batch and not has_glossary:
                self._process_single_entry(data, items, terms, val_model, glos_model)

    def _process_single_entry(
        self,
        entry: object,
        items: list[Any],
        terms: list[Any],
        val_model: type[T] | None = None,
        glos_model: type[object] | None = None,
    ) -> None:
        """Обрабатывает один элемент данных.

        Args:
            entry: Одиночный элемент данных для обработки.
            items: Коллекция основных элементов для наполнения.
            terms: Коллекция терминов глоссария.
            val_model: Класс модели для валидации основных элементов.
            glos_model: Класс модели для валидации терминов глоссария.
        """
        if not isinstance(entry, dict):
            return

        # 1. Если это термин глоссария (есть ключ 'term')
        if "term" in entry:
            self._add_item(entry, terms, glos_model)
            return

        # 2. Если это элемент перевода (есть 'id' или характерные поля)
        text_val = entry.get("translated_text") or entry.get("text")
        verify_val = entry.get("v") or entry.get("verification_prefix")

        # Если задана модель — работаем в строгом режиме валидации
        if val_model:
            if "id" in entry and text_val is not None and isinstance(text_val, str):
                item_payload = dict(entry)
                item_payload["id"] = str(entry["id"])
                item_payload["translated_text"] = text_val
                if verify_val is not None:
                    item_payload["v"] = str(verify_val)
                self._add_item(item_payload, items, val_model)
            elif "id" in entry:
                # Если есть ID, но нет текста — возможно, это вердикт вычитки или пустой перевод
                self._add_item(entry, items, val_model)
            else:
                # В остальных случаях пробуем, только если это не похоже на мусор
                if len(entry) > 0:
                    self._add_item(entry, items, val_model)
        else:
            # Универсальный режим (без модели) — берем всё
            if text_val is not None and isinstance(text_val, str):
                entry["translated_text"] = text_val
            items.append(entry)

    def _add_item(
        self,
        data: dict[str, object],
        collection: list[object],
        model: type[object] | None = None,
    ) -> None:
        """Добавляет элемент в коллекцию, опционально валидируя через модель.

        Args:
            data: Данные элемента в виде словаря.
            collection: Целевая коллекция для добавления.
            model: Опциональный класс модели для валидации.
        """
        if model:
            try:
                if hasattr(model, "model_validate"):
                    collection.append(model.model_validate(data))
                else:
                    collection.append(model(**data))
            except Exception as e:
                self._logger.warning("Ошибка валидации данных через модель %s: %s", model.__name__, e)
        else:
            collection.append(data)

    @staticmethod
    def _attempt_json_recovery(text: str) -> str:
        """Интеллектуально пытается восстановить поврежденный в конце JSON."""
        text = text.strip()
        while text and text[-1] in (",", ":", '"', "[", "{"):
            if (text[-1] == '"' and text.count('"') % 2 != 0) or text[-1] in (",", ":"):
                text = text[:-1].strip()
            else:
                break

        stack = []
        in_string = False
        escaped = False

        for char in text:
            if char == '"' and not escaped:
                in_string = not in_string
            if in_string:
                escaped = bool(char == "\\" and not escaped)
                continue

            if char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char == "}":
                if stack and stack[-1] == "}":
                    stack.pop()
            elif char == "]" and stack and stack[-1] == "]":
                stack.pop()

        if in_string:
            text += '"'

        while stack:
            text += stack.pop()

        return text
