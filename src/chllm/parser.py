"""
Компонент для робастного парсинга ответов от LLM.
Централизует логику очистки, исправления и разбора JSON данных.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, overload

if TYPE_CHECKING:
    from .triggers.registry import TriggerRegistry

from ._json_repair import (
    ToolCall,
    attempt_json_recovery,
    calculate_model_match_score,
    clean_markdown,
    extract_tool_calls,
    find_json_blocks,
    find_matching_list,
    sanitize_json_text,
)

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


class RobustLLMParser:
    """Универсальный парсер для обработки ответов от нейросетей."""

    def __init__(
        self,
        logger: Any | None = None,
        default_container_key: str = "batch",
        default_container_aliases: Sequence[str] = (
            "batch",
            "items",
            "data",
            "results",
            "options",
            "declensions",
            "words",
        ),
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
            else {"text": "translated_text", "verification_prefix": "v"}
        )

    def _get_candidate_keys(
        self,
        container_key: str | None = None,
        container_keys: str | Sequence[str] | None = None,
    ) -> list[str]:
        """Формирует упорядоченный список ключей-кандидатов для контейнера данных."""
        candidates: list[str] = []
        if container_key:
            candidates.append(container_key)
        if container_keys:
            if isinstance(container_keys, str):
                if container_keys not in candidates:
                    candidates.append(container_keys)
            else:
                for k in container_keys:
                    if k not in candidates:
                        candidates.append(k)

        for k in self._default_container_aliases:
            if k not in candidates:
                candidates.append(k)

        return candidates

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
            empty: dict[str, list[T | dict[str, Any]]] = {out_key: []}
            if include_empty_glossary or glossary_model:
                empty[glossary_key] = []
            return empty

        candidate_keys = self._get_candidate_keys(container_key, container_keys)

        merged_field_aliases = dict(self._field_aliases)
        if field_aliases is not None:
            merged_field_aliases.update(field_aliases)

        # 1. Извлекаем все сбалансированные JSON блоки из текста
        # Предварительно нормализуем типографические кавычки, чтобы строковые литералы
        # распознавались без искажения вложенных структур
        normalized_raw = re.sub(r"[«»“”„‟]", '"', raw_text)
        blocks = self._find_json_blocks(normalized_raw)

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
                # Пробуем санитизировать блок (кавычки, висячие запятые)
                sanitized_block = sanitize_json_text(block)
                try:
                    data = json.loads(sanitized_block)
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
                    continue
                except (json.JSONDecodeError, TypeError):
                    pass

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
                recovered = self._attempt_json_recovery(sanitized_block)
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

    @overload
    def parse_items(
        self,
        raw_text: str,
        validation_model: type[T],
        *,
        container_key: str | None = None,
        container_keys: str | Sequence[str] | None = None,
        field_aliases: Mapping[str, str] | None = None,
    ) -> list[T]: ...

    @overload
    def parse_items(
        self,
        raw_text: str,
        validation_model: None = None,
        *,
        container_key: str | None = None,
        container_keys: str | Sequence[str] | None = None,
        field_aliases: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]]: ...

    def parse_items(
        self,
        raw_text: str,
        validation_model: type[T] | None = None,
        *,
        container_key: str | None = None,
        container_keys: str | Sequence[str] | None = None,
        field_aliases: Mapping[str, str] | None = None,
    ) -> Any:
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

    @overload
    def parse_stream(
        self,
        stream: AsyncIterable[str],
        validation_model: type[T],
        *,
        container_key: str | None = None,
        container_keys: str | Sequence[str] | None = None,
        field_aliases: Mapping[str, str] | None = None,
    ) -> AsyncIterator[T]: ...

    @overload
    def parse_stream(
        self,
        stream: AsyncIterable[str],
        validation_model: None = None,
        *,
        container_key: str | None = None,
        container_keys: str | Sequence[str] | None = None,
        field_aliases: Mapping[str, str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]: ...

    async def parse_stream(
        self,
        stream: AsyncIterable[str],
        validation_model: type[T] | None = None,
        *,
        container_key: str | None = None,
        container_keys: str | Sequence[str] | None = None,
        field_aliases: Mapping[str, str] | None = None,
    ) -> AsyncIterator[Any]:
        """Асинхронно парсит поток текста от LLM, выдавая готовые элементы по мере поступления.

        Извлекает и валидирует завершенные объекты JSON в реальном времени,
        не дожидаясь завершения всего ответа нейросети.

        Args:
            stream: Асинхронный итератор чанков текста (токенов).
            validation_model: Опциональная Pydantic-модель для элементов.
            container_key: Имя целевого контейнера.
            container_keys: Список допустимых ключей контейнера или одиночный ключ.
            field_aliases: Маппинг алиасов полей.

        Yields:
            Валидированные объекты модели (или словари) по мере их завершения в потоке.
        """
        accumulated_text = ""
        yielded_count = 0

        async for chunk in stream:
            accumulated_text += chunk
            # Пробуем распарсить накопленный текст
            items = self.parse_items(
                accumulated_text,
                validation_model=validation_model,
                container_key=container_key,
                container_keys=container_keys,
                field_aliases=field_aliases,
            )

            # Если появились новые завершенные элементы
            if len(items) > yielded_count:
                for new_item in items[yielded_count:]:
                    yield new_item
                yielded_count = len(items)

    @staticmethod
    def _find_json_blocks(text: str) -> list[str]:
        """Находит все сбалансированные JSON-блоки ({...} или [...]) в тексте."""
        return find_json_blocks(text)

    def parse_raw(self, raw_text: str) -> object:
        """Просто парсит JSON с санитизацией и восстановлением, возвращая чистый Python объект.

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
            sanitized = sanitize_json_text(clean)
            try:
                return json.loads(sanitized)
            except json.JSONDecodeError:
                recovered = self._attempt_json_recovery(sanitized)
                return json.loads(recovered)

    @staticmethod
    def sanitize_conversational_text(text: str) -> str:
        """Очищает разговорный текст ответа от LLM перед парсингом JSON.

        Удаляет markdown-разметку, фразы вежливости веб-чатов,
        нормализует типографические кавычки и висячие запятые.

        Args:
            text: Исходный ответ от нейросети.

        Returns:
            Очищенный текст JSON.
        """
        if not text:
            return ""

        clean = clean_markdown(text)
        return sanitize_json_text(clean)

    @overload
    def parse_conversational_response(
        self,
        raw_text: str,
        validation_model: type[T],
        *,
        container_key: str | None = None,
        container_keys: str | Sequence[str] | None = None,
        field_aliases: Mapping[str, str] | None = None,
    ) -> list[T]: ...

    @overload
    def parse_conversational_response(
        self,
        raw_text: str,
        validation_model: None = None,
        *,
        container_key: str | None = None,
        container_keys: str | Sequence[str] | None = None,
        field_aliases: Mapping[str, str] | None = None,
    ) -> Any: ...

    def parse_conversational_response(
        self,
        raw_text: str,
        validation_model: type[T] | None = None,
        *,
        container_key: str | None = None,
        container_keys: str | Sequence[str] | None = None,
        field_aliases: Mapping[str, str] | None = None,
    ) -> Any:
        """Парсит неструктурированный или разговорный ответ веб-чата/модели.

        Устойчив к вежливости, преамбулам/постамбулам, типографическим кавычкам («», “”),
        висячим запятым, отсутствующим внешним скобкам списков ([...]) и поврежденному JSON.

        Args:
            raw_text: Текст ответа от веб-модели или чата.
            validation_model: Опциональная Pydantic-модель для валидации элементов.
            container_key: Имя целевого контейнера (по умолчанию 'batch').
            container_keys: Список допустимых ключей контейнера или одиночный ключ.
            field_aliases: Маппинг алиасов полей.

        Returns:
            Список валидированных объектов модели (или список/словарь распарсенных данных).
        """
        if not raw_text or not raw_text.strip():
            return []

        out_key = container_key or self._default_container_key
        candidate_keys = self._get_candidate_keys(container_key, container_keys)

        # 1. Попытка через стандартный parse_items (с санитизацией блоков)
        if items := self.parse_items(
            raw_text,
            validation_model=validation_model,
            container_key=out_key,
            container_keys=candidate_keys,
            field_aliases=field_aliases,
        ):
            return items

        # 2. Санитизация разговорного текста
        clean_text = self.sanitize_conversational_text(raw_text)

        # Попытка парсинга очищенного текста
        parsed_data = None
        try:
            parsed_data = self.parse_raw(clean_text)
        except Exception:
            # Если это список объектов через запятую без скобок [...]
            with contextlib.suppress(Exception):
                parsed_data = self.parse_raw(f"[{clean_text}]")

        # Если распарсить единым куском не удалось — regex-поиск всех объектов {...}
        if parsed_data is None:
            regex_items: list[dict[str, Any]] = []
            for match in re.finditer(r"\{[^{}]*\}", clean_text, re.DOTALL):
                with contextlib.suppress(Exception):
                    if isinstance(obj := self.parse_raw(match.group(0)), dict):
                        regex_items.append(obj)
            if regex_items:
                parsed_data = regex_items

        if parsed_data is None:
            return []

        # Извлечение элементов из parsed_data
        raw_items: list[Any] = []
        if isinstance(parsed_data, list):
            raw_items = parsed_data
        elif isinstance(parsed_data, dict):
            # Проверяем контейнерные ключи
            for k in candidate_keys:
                if k in parsed_data and isinstance(parsed_data[k], list):
                    raw_items = parsed_data[k]
                    break
            else:
                # Ищем любой список внутри словаря
                for val in parsed_data.values():
                    if isinstance(val, list):
                        raw_items = val
                        break
                else:
                    # Одиночный объект
                    raw_items = [parsed_data]

        # Валидация через модель при наличии
        if validation_model:
            validated: list[Any] = []
            for item in raw_items:
                if isinstance(item, dict):
                    self._add_item(item, validated, validation_model)
            return validated

        return raw_items if raw_items else parsed_data

    @staticmethod
    def parse_tool_calls(
        text: str,
        triggers: TriggerRegistry | None = None,
    ) -> list[ToolCall]:
        """Ищет и парсит вызовы инструментов (JSON блоки и текстовые триггеры) в тексте.

        Args:
            text: Сырой текст ответа ИИ.
            triggers: Опциональный реестр текстовых fallback-триггеров.

        Returns:
            Дедуплицированный список объектов ToolCall.
        """
        return extract_tool_calls(text, triggers)

    @staticmethod
    def _clean_markdown(text: str) -> str:
        return clean_markdown(text)

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
        glos_list = data.get(glossary_key) or data.get("suggested_glossary_terms")
        if isinstance(glos_list, list):
            has_glossary = True
            for term_data in glos_list:
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
        """Ищет подходящий список объектов внутри словаря по полям модели."""
        return find_matching_list(data, val_model, glossary_keys)

    def _calculate_model_match_score(self, entries: list[dict[str, Any]], model: type[Any]) -> int:
        """Вычисляет оценку соответствия списка словарей модели Pydantic."""
        return calculate_model_match_score(entries, model)

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
        return attempt_json_recovery(text)
