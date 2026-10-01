"""
Внутренние утилиты для очистки, поиска и восстановления поврежденного JSON.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from .triggers.registry import TriggerRegistry


class ToolCall(BaseModel):
    """Модель вызова инструмента ИИ."""

    name: str
    args: dict[str, Any] = {}


def find_json_blocks(text: str) -> list[str]:
    """Находит все сбалансированные JSON-блоки ({...} или [...]) в тексте.

    Args:
        text: Входной текст от LLM.

    Returns:
        Список найденных JSON-строк.
    """
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
        i += 1
    return blocks


def clean_markdown(text: str) -> str:
    """Извлекает чистое содержимое JSON из текста (игнорируя болтовню нейросети).

    Args:
        text: Сырой текст ответа от LLM.

    Returns:
        Очищенный JSON текст.
    """
    if not text:
        return ""

    # 1. Если есть блок кода markdown - извлекаем ТОЛЬКО его содержимое
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. Если блоков маркдауна нет, жестко обрезаем текст по границам JSON
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


def sanitize_json_text(text: str) -> str:
    """Очищает и нормализует строку JSON от разговорного мусора и частых ошибок веб-моделей.

    - Заменяет типографические (фигурные/русские) кавычки (« », “ ”, „ ‟) на ASCII-кавычки.
    - Удаляет висячие запятые перед закрывающими скобками (, ] -> ] и , } -> }).
    - Удаляет лишние знаки препинания по краям (запятые, точки с запятой).
    - Преобразует Python-стиль одинарных кавычек в ключах/строках в валидный JSON.

    Args:
        text: Исходный текст JSON.

    Returns:
        Санитизированная строка JSON.
    """
    if not text:
        return ""

    cleaned = text.strip().rstrip(",; \t\n\r")

    # 1. Замена типографических кавычек на стандартные прямые двойные кавычки
    cleaned = re.sub(r"[«»“”„‟]", '"', cleaned)

    # 2. Удаление висячих запятых перед закрывающими фигурными и квадратными скобками
    cleaned = re.sub(r",\s*([\]\}])", r"\1", cleaned)

    # 3. Нормализация одинарных кавычек в стиле Python dict: {'key': 'value'}
    if "'" in cleaned and '"' not in cleaned:
        cleaned = re.sub(
            r"(?<=[\{\[\,\:\s])'([^'\\]*(?:\\.[^'\\]*)*)'(?=[\}\]\,\:\s])",
            r'"\1"',
            cleaned,
        )

    return cleaned


def attempt_json_recovery(text: str) -> str:
    """Интеллектуально пытается восстановить поврежденный в конце JSON.

    Args:
        text: Оборванная строка JSON.

    Returns:
        Восстановленная строка JSON со сбалансированными кавычками и скобками.
    """
    text = text.strip()
    text = re.sub(r",\s*([\]\}])", r"\1", text)
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


def calculate_model_match_score(entries: list[dict[str, Any]], model: type[Any]) -> int:
    """Вычисляет оценку соответствия списка словарей модели Pydantic.

    Args:
        entries: Список словарей.
        model: Класс Pydantic-модели.

    Returns:
        Счетчик совпавших полей.
    """
    model_fields: set[str] = set()
    model_aliases: set[str] = set()

    if hasattr(model, "model_fields"):
        for name, field_info in model.model_fields.items():
            model_fields.add(name)
            if field_info.alias:
                model_aliases.add(field_info.alias)
            if getattr(field_info, "validation_alias", None):
                val_alias = field_info.validation_alias
                if isinstance(val_alias, str):
                    model_aliases.add(val_alias)
    elif hasattr(model, "__fields__"):
        for name, field_info in model.__fields__.items():
            model_fields.add(name)
            if field_info.alias:
                model_aliases.add(field_info.alias)
    else:
        return 0

    target_keys = model_fields | model_aliases
    total_score = 0
    for entry in entries[:3]:
        matched = set(entry.keys()) & target_keys
        total_score += len(matched)

    return total_score


def find_matching_list(
    data: dict[str, Any],
    val_model: type[Any] | None,
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
            dict_entries = [e for e in value if isinstance(e, dict)]
            if not dict_entries:
                continue

            if val_model:
                match_score = calculate_model_match_score(dict_entries, val_model)
                if match_score > 0:
                    candidate_lists.append((match_score, value))
            else:
                candidate_lists.append((1, value))

    if candidate_lists:
        candidate_lists.sort(key=lambda x: x[0], reverse=True)
        return candidate_lists[0][1]

    return None


def extract_tool_calls(
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
    if not text:
        return []

    results: list[ToolCall] = []

    blocks = find_json_blocks(text)
    for block in blocks:
        try:
            data = json.loads(block)
            if not isinstance(data, dict):
                continue

            for wrapper in ("tool_use", "tool", "call", "function"):
                if wrapper in data and isinstance(data[wrapper], dict):
                    inner = data[wrapper]
                    if "name" in inner:
                        data = inner
                        break

            if "name" not in data or not isinstance(data["name"], str):
                continue

            name = data["name"]
            args = {}

            for arg_field in ("args", "parameters", "arguments", "params"):
                if arg_field in data and isinstance(data[arg_field], dict):
                    args = data[arg_field]
                    break
            else:
                excluded_keys = {"name", "tool_use", "tool", "call", "function"}
                args = {k: v for k, v in data.items() if k not in excluded_keys}

            results.append(ToolCall(name=name, args=args))
        except (json.JSONDecodeError, ValueError):
            continue

    if triggers is not None:
        trigger_calls = triggers.parse_triggers(text)
        results.extend(trigger_calls)

    seen: set[tuple[str, str]] = set()
    deduped: list[ToolCall] = []
    for call in results:
        key = (call.name, json.dumps(call.args, sort_keys=True, ensure_ascii=False, default=str))
        if key not in seen:
            seen.add(key)
            deduped.append(call)

    return deduped
