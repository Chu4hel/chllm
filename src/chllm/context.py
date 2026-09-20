"""
Модуль для управления контекстом и историей в батч-запросах к LLM.
Реализует различные стратегии формирования предыстории для элементов.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ContextItem:
    """Элемент данных с привязанным к нему контекстом.

    Attributes:
        data: Исходные данные элемента.
        context: Список строк предыстории.
    """

    data: Any
    context: list[str]


class ContextBuilder:
    """Универсальный строитель контекста для последовательных данных."""

    def __init__(self, strategy: str = "chain"):
        """Инициализирует строитель.

        Args:
            strategy: Стратегия формирования контекста ("chain", "full").
        """
        self._strategy = strategy

    def build(
        self, items: list[Any], initial_history: list[str] | None = None, formatter: Callable[[Any], str] | None = None
    ) -> list[ContextItem]:
        """Формирует контекст для списка элементов.

        Args:
            items: Список элементов для обработки.
            initial_history: Начальная история (для первого элемента).
            formatter: Функция для преобразования элемента в строку контекста
                       для последующих элементов.

        Returns:
            Список объектов ContextItem.
        """
        if not items:
            return []

        # Используем стандартный строковый форматтер, если не задан кастомный
        if formatter is None:
            formatter = self._default_formatter

        result = []
        history = initial_history or []

        if self._strategy == "chain":
            for i, item in enumerate(items):
                if i == 0:
                    # Первый элемент получает всю начальную историю
                    context_for_item = history
                else:
                    # Последующие получают только предыдущий элемент
                    prev_item = items[i - 1]
                    context_for_item = [formatter(prev_item)]

                result.append(ContextItem(data=item, context=context_for_item))

        elif self._strategy == "full":
            # Стратегия "Полный контекст": каждый получает всю предыдущую историю
            current_full_history = list(history)
            for item in items:
                result.append(ContextItem(data=item, context=list(current_full_history)))
                current_full_history.append(formatter(item))

        return result

    @staticmethod
    def _default_formatter(item: Any) -> str:
        """Стандартный форматтер: преобразует данные в строку."""
        return str(item)
