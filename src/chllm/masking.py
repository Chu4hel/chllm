"""
Модуль для маскирования (защиты) частей текста от изменений со стороны LLM.
Используется для защиты переменных, тегов и специальных символов.
"""

import re
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


class ContentMasker:
    """Универсальный компонент для маскирования текста на основе регулярных выражений."""

    def __init__(
        self,
        patterns: list[str] | None = None,
        placeholder_prefix: str = "VAR",
        placeholder_template: str = "[[[{prefix}_{index}]]]",
    ):
        """Инициализирует маскер.

        Args:
            patterns: Список регулярных выражений для поиска защищаемых частей.
            placeholder_prefix: Префикс для плейсхолдеров (напр. "VAR", "TAG").
            placeholder_template: Шаблон формирования плейсхолдера.
        """
        self._patterns = patterns or []
        self._prefix = placeholder_prefix
        self._template = placeholder_template

    def mask(
        self,
        text: str,
        extra_patterns: list[str] | None = None,
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

        all_patterns = self._patterns + (extra_patterns or [])
        if not all_patterns:
            return MaskedResult(masked_text=text)

        exclude_set = set(excluded_values or [])
        mapping: dict[str, str] = {}
        masked_text = text

        # Объединяем паттерны в одно регулярное выражение для поиска всех вхождений сразу.
        # Сортируем паттерны по длине в обратном порядке, чтобы длинные вхождения
        # (напр. \\n) имели приоритет над короткими (напр. \n).
        sorted_patterns = sorted(all_patterns, key=len, reverse=True)
        combined_regex = re.compile("|".join(f"({p})" for p in sorted_patterns))

        def replace_match(match: re.Match) -> str:
            # Извлекаем реально совпавшую строку (первая не-None группа)
            original_value = match.group(0)

            # Если значение находится в списке исключений, оставляем его без маскировки
            if original_value in exclude_set:
                return original_value

            # Проверяем, не маскировали ли мы это значение ранее
            # (для экономии индексов и единообразия)
            for p, v in mapping.items():
                if v == original_value:
                    return p

            # Создаем новый плейсхолдер
            index = len(mapping)
            placeholder = self._template.format(prefix=self._prefix, index=index)
            mapping[placeholder] = original_value
            return placeholder

        # Выполняем замену всех найденных вхождений
        masked_text = combined_regex.sub(replace_match, masked_text)

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
        # Заменяем в обратном порядке (плейсхолдеры на оригиналы)
        for placeholder, original_value in mapping.items():
            demasked_text = demasked_text.replace(placeholder, original_value)

        return demasked_text
