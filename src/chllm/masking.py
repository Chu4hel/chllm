"""Модуль для маскирования (защиты) частей текста от изменений со стороны LLM.

Используется для защиты переменных, тегов разметки, специальных символов
и конструкций интерполяции от искажения нейросетью в процессе перевода или генерации.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


@dataclass
class MaskedResult:
    """Результат операции маскирования.

    Attributes:
        masked_text: Текст, в котором целевые части заменены на плейсхолдеры.
        mapping: Словарь соответствия плейсхолдеров оригинальным значениям.
    """

    masked_text: str
    mapping: dict[str, str] = field(default_factory=dict)


class MaskingPreset(tuple[str, ...]):
    """Неизменяемый набор шаблонов маскирования с поддержкой побитового объединения (|)."""

    def __or__(self, other: Iterable[str]) -> MaskingPreset:
        """Объединяет два набора шаблонов без дубликатов с сохранением порядка.

        Args:
            other: Другой набор шаблонов или пресет.

        Returns:
            Новый комбинированный MaskingPreset.
        """
        combined = list(self)
        for pattern in other:
            if pattern not in combined:
                combined.append(pattern)
        return MaskingPreset(combined)

    def __ror__(self, other: Iterable[str]) -> MaskingPreset:
        """Правостороннее объединение (для работы синтаксиса list | MaskingPreset).

        Args:
            other: Исходный набор шаблонов.

        Returns:
            Новый комбинированный MaskingPreset.
        """
        combined: list[str] = []
        for pattern in other:
            if pattern not in combined:
                combined.append(pattern)
        for pattern in self:
            if pattern not in combined:
                combined.append(pattern)
        return MaskingPreset(combined)

    def __add__(self, other: Iterable[str]) -> MaskingPreset:  # type: ignore[override]
        """Сложение наборов шаблонов (псевдоним для оператора |).

        Args:
            other: Другой набор шаблонов.

        Returns:
            Новый комбинированный MaskingPreset.
        """
        return self | other


class MaskingPresets:
    """Коллекция готовых пресетов регулярных выражений для ContentMasker."""

    BRACKETS = MaskingPreset((r"\[.+?\]",))
    """Квадратные скобки интерполяции переменных (напр. [player_name], [points])."""

    BRACES = MaskingPreset((r"\{.+?\}",))
    """Фигурные скобки тегов стилей и форматирования (напр. {b}...{/b}, {color=#f00})."""

    PRINTF = MaskingPreset((r"%\([a-zA-Z0-9_]+\)[a-zA-Z%]|%[a-zA-Z%]",))
    """Классическая printf / Python-интерполяция (напр. %s, %d, %(user)s, %%)."""

    NEWLINES = MaskingPreset((r"\\r\\n", r"\\n", r"\r\n", r"\n"))
    """Символы переноса строк (\n, \r\n) и их экранированные строковые литералы (\\n, \\r\\n)."""

    HTML = MaskingPreset((r"<!--[\s\S]*?-->", r"<[^>]+>", r"&[a-zA-Z0-9#]+;"))
    """HTML теги, комментарии и сущности (&nbsp;, &lt; и т.д.)."""

    MARKDOWN_LINKS = MaskingPreset((r"\[.+?\]\(.+?\)",))
    """Markdown ссылки вида [текст](url)."""

    CODE_BLOCKS = MaskingPreset((r"```[\s\S]*?```", r"`[^`]+`"))
    """Блоки кода Markdown (fenced code blocks и инлайн код)."""

    URLS = MaskingPreset((r"https?://[^\s<>\"']+",))
    """Веб-ссылки (HTTP/HTTPS URL)."""

    # Составные пресеты
    RENPY = BRACKETS | BRACES | PRINTF
    """Полный набор синтаксических тегов и интерполяций Ren'Py (скобки, фигурные теги, printf)."""

    ALL_INTERPOLATIONS = BRACKETS | BRACES | PRINTF
    """Общий набор всех распространенных шаблонов интерполяции переменных."""


class ContentMasker:
    """Универсальный компонент для маскирования текста на основе регулярных выражений."""

    Presets = MaskingPresets
    """Ссылка на коллекцию пресетов MaskingPresets для удобного доступа."""

    def __init__(
        self,
        patterns: Sequence[str] | None = None,
        placeholder_prefix: str = "VAR",
        placeholder_template: str = "[[[{prefix}_{index}]]]",
    ) -> None:
        """Инициализирует маскер.

        Args:
            patterns: Список регулярных выражений или пресет MaskingPresets.
            placeholder_prefix: Префикс для плейсхолдеров (напр. 'VAR', 'TAG').
            placeholder_template: Шаблон формирования плейсхолдера.
        """
        self._patterns = list(patterns) if patterns is not None else []
        self._prefix = placeholder_prefix
        self._template = placeholder_template

    def mask(
        self,
        text: str,
        extra_patterns: Sequence[str] | None = None,
        excluded_values: set[str] | list[str] | None = None,
    ) -> MaskedResult:
        """Маскирует все вхождения паттернов в тексте.

        Args:
            text: Исходный текст.
            extra_patterns: Дополнительные паттерны только для этого вызова.
            excluded_values: Набор значений/паттернов, которые не должны маскироваться.

        Returns:
            Объект MaskedResult с результатом и картой маскировки.
        """
        if not text:
            return MaskedResult(masked_text=text)

        all_patterns = self._patterns + list(extra_patterns or [])
        if not all_patterns:
            return MaskedResult(masked_text=text)

        # Удаляем дубликаты с сохранением исходного порядка
        seen: set[str] = set()
        unique_patterns: list[str] = []
        for p in all_patterns:
            if p not in seen:
                seen.add(p)
                unique_patterns.append(p)

        exclude_set = set(excluded_values or [])
        mapping: dict[str, str] = {}

        # Сортируем паттерны по длине в обратном порядке, чтобы длинные вхождения
        # (напр. \\n) имели приоритет над короткими (напр. \n).
        sorted_patterns = sorted(unique_patterns, key=len, reverse=True)
        combined_regex = re.compile("|".join(f"({p})" for p in sorted_patterns))

        def replace_match(match: re.Match[str]) -> str:
            original_value = match.group(0)

            if original_value in exclude_set:
                return original_value

            for p, v in mapping.items():
                if v == original_value:
                    return p

            index = len(mapping)
            placeholder = self._template.format(prefix=self._prefix, index=index)
            mapping[placeholder] = original_value
            return placeholder

        masked_text = combined_regex.sub(replace_match, text)
        return MaskedResult(masked_text=masked_text, mapping=mapping)

    def demask(self, masked_text: str, mapping: dict[str, str]) -> str:
        """Восстанавливает оригинальный текст из маскированного.

        Args:
            masked_text: Текст с плейсхолдерами.
            mapping: Карта маскировки (из MaskedResult).

        Returns:
            Восстановленный текст.
        """
        if not masked_text or not mapping:
            return masked_text

        demasked_text = masked_text
        for placeholder, original_value in mapping.items():
            demasked_text = demasked_text.replace(placeholder, original_value)

        return demasked_text
