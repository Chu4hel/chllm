"""
Компонент для робастного парсинга ответов от LLM.
Централизует логику очистки, исправления и разбора JSON данных.
"""

import json
import re
from collections.abc import Mapping, Sequence
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

    def __init__(
        self,
        logger: Any | None = None,
        default_container_key: str = "batch",
        default_container_aliases: Sequence[str] = ("batch", "items", "data", "results"),
        field_aliases: Mapping[str, str] | None = None,
    ):
        """Инициализирует парсер ответов от LLM.

        Args:
            logger: Опциональный логгер.
            default_container_key: Имя ключа для основных элементов по умолчанию.
            default_container_aliases: Список стандартных имен контейнеров для поиска.
            field_aliases: Опциональный маппинг алиасов полей, например {"text": "translated_text"}.
        """
        self._logger = logger or setup_logger(__name__)
        self._default_container_key = default_container_key
        self._default_container_aliases = tuple(default_container_aliases)
        self._field_aliases = (
            dict(field_aliases)
            if field_aliases is not None
            else {
                "text": "translated_text",
                "verification_prefix": "v",
            }
        )

    def parse(
        self,
        raw_text: str,
        validation_model: type[T] | None = None,
        glossary_model: type[Any] | None = None,
        *,
        container_key: str | None = None,
        container_keys: str | Sequence[str] | None = None,
        glossary_key: str = "suggested_glossary_terms",
        field_aliases: Mapping[str, str] | None = None,
        include_empty_glossary: bool = True,
    ) -> dict[str, list[T | dict[str, Any]]]:
        """Парсит сырой текст ответа ИИ в структурированный словарь.

        Args:
            raw_text: Текст ответа от ИИ.
            validation_model: Опциональная Pydantic-модель для основных элементов.
            glossary_model: Опциональная Pydantic-модель для терминов глоссария.
            container_key: Имя возвращаемого ключа или явное имя контейнера.
            container_keys: Список допустимых ключей контейнера или одиночный ключ.
            glossary_key: Имя ключа глоссария (по умолчанию 'suggested_glossary_terms').
            field_aliases: Маппинг алиасов полей для нормализации перед валидацией.
            include_empty_glossary: Включать ли ключ глоссария в результат, если он пуст.

        Returns:
            Dict[str, List]: Словарь с ключом контейнера (по умолчанию 'batch' или container_key)
            и глоссарием.
        """
        out_key = container_key or self._default_container_key

        if not raw_text or not raw_text.strip():
            result_empty: dict[str, list[T | dict[str, Any]]] = {out_key: []}
            if include_empty_glossary or glossary_model:
                result_empty[glossary_key] = []
            return result_empty

        # Вычисляем список ключей-кандидатов для контейнера
        candidate_keys: list[str] = []
        if container_key:
            candidate_keys.append(container_key)
        if container_keys:
            if isinstance(container_keys, str):
                if container_keys not in candidate_keys:
                    candidate_keys.append(container_keys)
            else:
                for k in container_keys:
                    if k not in candidate_keys:
                        candidate_keys.append(k)

        # Добавляем стандартные алиасы
        for k in self._default_container_aliases:
            if k not in candidate_keys:
                candidate_keys.append(k)

        merged_field_aliases = dict(self._field_aliases)
        if field_aliases is not None:
            merged_field_aliases.update(field_aliases)

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
                    container_keys=candidate_keys,
                    glossary_key=glossary_key,
                    field_aliases=merged_field_aliases,
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
                        container_keys=candidate_keys,
                        glossary_key=glossary_key,
                        field_aliases=merged_field_aliases,
                    )

                    if len(validated_items) > items_before or len(suggested_glossary_terms) > terms_before:
                        self._logger.info(
                            "Успешно исправлен и распарсен усеченный JSON: '%s...'",
                            block[:50],
                        )
                except json.JSONDecodeError:
                    continue

        result: dict[str, list[T | dict[str, Any]]] = {out_key: validated_items}
        if include_empty_glossary or suggested_glossary_terms or glossary_model:
            result[glossary_key] = suggested_glossary_terms

        return result

    def parse_items(
        self,
        raw_text: str,
        validation_model: type[T] | None = None,
        *,
        container_key: str | None = None,
        container_keys: str | Sequence[str] | None = None,
        field_aliases: Mapping[str, str] | None = None,
    ) -> list[T | dict[str, Any]]:
        """Парсит ответ нейросети и сразу возвращает список элементов.

        Удобный метод-обертка для прямого получения списка валидированных сущностей
        без необходимости извлекать их по ключу словаря.

        Args:
            raw_text: Текст ответа от ИИ.
            validation_model: Опциональная Pydantic-модель для элементов.
            container_key: Имя целевого контейнера.
            container_keys: Список допустимых ключей контейнера или одиночный ключ.
            field_aliases: Маппинг алиасов полей.

        Returns:
            Список валидированных объектов модели (или словарей).
        """
        out_key = container_key or self._default_container_key
        parsed = self.parse(
            raw_text,
            validation_model=validation_model,
            container_key=out_key,
            container_keys=container_keys,
            field_aliases=field_aliases,
            include_empty_glossary=False,
        )
        return parsed.get(out_key, [])

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
        container_keys: Sequence[str] = ("batch", "items", "data", "results"),
        glossary_key: str = "suggested_glossary_terms",
        field_aliases: Mapping[str, str] | None = None,
    ) -> None:
        """Разбирает распарсенный JSON и наполняет коллекции.

        Args:
            data: Распарсенные данные (список или словарь).
            items: Коллекция основных элементов для наполнения.
            terms: Коллекция терминов глоссария.
            val_model: Класс модели для валидации основных элементов.
            glos_model: Класс модели для валидации терминов глоссария.
            container_keys: Список приоритетных имен контейнеров.
            glossary_key: Имя ключа глоссария.
            field_aliases: Маппинг алиасов полей.
        """
        if isinstance(data, list):
            for entry in data:
                self._process_single_entry(entry, items, terms, val_model, glos_model, field_aliases)
            return

        if not isinstance(data, dict):
            return

        # 1. Проверяем глоссарий
        has_glossary = False
        if glossary_key in data and isinstance(data[glossary_key], list):
            has_glossary = True
            for term_data in data[glossary_key]:
                if isinstance(term_data, dict) and "term" in term_data:
                    self._add_item(term_data, terms, glos_model)
        elif "suggested_glossary_terms" in data and isinstance(data["suggested_glossary_terms"], list):
            has_glossary = True
            for term_data in data["suggested_glossary_terms"]:
                if isinstance(term_data, dict) and "term" in term_data:
                    self._add_item(term_data, terms, glos_model)

        # 2. Поиск основных элементов по ключам контейнера
        found_container_key = None
        for key in container_keys:
            if key in data and isinstance(data[key], list):
                found_container_key = key
                break

        if found_container_key is not None:
            for entry in data[found_container_key]:
                self._process_single_entry(entry, items, terms, val_model, glos_model, field_aliases)
            return

        # 3. Fallback: Автоматический поиск списка по сигнатуре модели (Zero-Config fallback)
        best_list = self._find_matching_list(data, val_model, glossary_keys={glossary_key, "suggested_glossary_terms"})
        if best_list is not None:
            for entry in best_list:
                self._process_single_entry(entry, items, terms, val_model, glos_model, field_aliases)
            return

        # 4. Если это одиночный объект (не контейнер со списком)
        if not has_glossary:
            self._process_single_entry(data, items, terms, val_model, glos_model, field_aliases)

    def _find_matching_list(
        self,
        data: dict[str, Any],
        val_model: type[T] | None,
        glossary_keys: set[str],
    ) -> list[Any] | None:
        """Ищет подходящий список объектов внутри словаря по полям модели.

        Args:
            data: Словарь данных верхнего уровня.
            val_model: Опциональная модель валидации.
            glossary_keys: Множество ключей глоссария для исключения.

        Returns:
            Найденный список объектов или None.
        """
        candidate_lists: list[tuple[int, list[Any]]] = []

        for key, value in data.items():
            if key in glossary_keys:
                continue
            if isinstance(value, list) and value:
                # Проверяем, что элементы - словари
                dict_entries = [e for e in value if isinstance(e, dict)]
                if not dict_entries:
                    continue

                if val_model:
                    match_score = self._calculate_model_match_score(dict_entries, val_model)
                    if match_score > 0:
                        candidate_lists.append((match_score, value))
                else:
                    # Без модели берем первый непустой список словарей
                    candidate_lists.append((1, value))

        if candidate_lists:
            # Сортируем по убыванию соответствия
            candidate_lists.sort(key=lambda x: x[0], reverse=True)
            return candidate_lists[0][1]

        return None

    def _calculate_model_match_score(self, entries: list[dict[str, Any]], model: type[Any]) -> int:
        """Вычисляет оценку соответствия списка словарей модели Pydantic."""
        model_fields: set[str] = set()
        model_aliases: set[str] = set()

        if hasattr(model, "model_fields"):
            # Pydantic v2
            for name, field_info in model.model_fields.items():
                model_fields.add(name)
                if field_info.alias:
                    model_aliases.add(field_info.alias)
                if getattr(field_info, "validation_alias", None):
                    val_alias = field_info.validation_alias
                    if isinstance(val_alias, str):
                        model_aliases.add(val_alias)
        elif hasattr(model, "__fields__"):
            # Pydantic v1
            for name, field_info in model.__fields__.items():
                model_fields.add(name)
                if field_info.alias:
                    model_aliases.add(field_info.alias)
        else:
            return 0

        target_keys = model_fields | model_aliases

        # Проверяем до 3 первых записей
        total_score = 0
        samples = entries[:3]
        for entry in samples:
            entry_keys = set(entry.keys())
            matched = entry_keys & target_keys
            total_score += len(matched)

        return total_score

    def _process_single_entry(
        self,
        entry: object,
        items: list[Any],
        terms: list[Any],
        val_model: type[T] | None = None,
        glos_model: type[object] | None = None,
        field_aliases: Mapping[str, str] | None = None,
    ) -> None:
        """Обрабатывает один элемент данных.

        Args:
            entry: Одиночный элемент данных для обработки.
            items: Коллекция основных элементов для наполнения.
            terms: Коллекция терминов глоссария.
            val_model: Класс модели для валидации основных элементов.
            glos_model: Класс модели для валидации терминов глоссария.
            field_aliases: Маппинг алиасов полей.
        """
        if not isinstance(entry, dict):
            return

        # 1. Если это термин глоссария (есть ключ 'term')
        if "term" in entry and glos_model:
            self._add_item(entry, terms, glos_model)
            return

        item_payload = dict(entry)

        # Применяем field_aliases (например, text -> translated_text)
        if field_aliases:
            for src_field, dst_field in field_aliases.items():
                if src_field in item_payload and dst_field not in item_payload:
                    item_payload[dst_field] = item_payload[src_field]

        # Для обратной совместимости: если в модели есть str id, приводим id к str
        if "id" in item_payload and val_model:
            item_payload["id"] = str(item_payload["id"])

        # Валидация / добавление
        if val_model:
            if len(item_payload) > 0:
                self._add_item(item_payload, items, val_model)
        else:
            items.append(item_payload)

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
