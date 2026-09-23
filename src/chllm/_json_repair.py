"""
Внутренние утилиты для очистки, поиска и восстановления поврежденного JSON.
"""

import re
from typing import Any

from pydantic import BaseModel


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


def attempt_json_recovery(text: str) -> str:
    """Интеллектуально пытается восстановить поврежденный в конце JSON.

    Args:
        text: Оборванная строка JSON.

    Returns:
        Восстановленная строка JSON со сбалансированными кавычками и скобками.
    """
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
