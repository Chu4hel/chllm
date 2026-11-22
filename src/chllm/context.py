"""
Модуль для управления контекстом и историей в батч-запросах к LLM.
Реализует различные стратегии формирования предыстории для элементов.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .chunker import ChunkingConfig, TextChunker, TokenChunker


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


def compress_history_messages(
    history_repr: list[str],
    keep_recent: int = 4,
    max_old_chars: int = 300,
    preview_chars: int = 200,
) -> list[str]:
    """Сжимает длинные промежуточные шаги диалога, сохраняя первое и последние сообщения.

    Args:
        history_repr: Список отформатированных сообщений истории.
        keep_recent: Количество последних сообщений, оставляемых в полном виде.
        max_old_chars: Порог длины для сжатия старого сообщения.
        preview_chars: Сколько первых символов оставлять при сжатии.

    Returns:
        Новый список сообщений со сжатыми старыми наблюдениями.
    """
    if len(history_repr) <= keep_recent + 1:
        return list(history_repr)

    cutoff = len(history_repr) - keep_recent
    compressed: list[str] = [history_repr[0]]
    for idx in range(1, cutoff):
        item = history_repr[idx]
        if len(item) > max_old_chars:
            compressed.append(
                f"{item[:preview_chars].rstrip()}...\n[...сжато для экономии контекста, всего {len(item)} симв.]"
            )
        else:
            compressed.append(item)
    compressed.extend(history_repr[cutoff:])
    return compressed


def build_step_limit_instruction(max_steps: int | None) -> str:
    """Формирует явную инструкцию для LLM об ограничении шагов агента.

    Args:
        max_steps: Заданный пользователем/конфигурацией лимит (0 или None — без лимита).

    Returns:
        Строка инструкции для включения в системный промпт или задачу агента.
    """
    if max_steps is None or max_steps <= 0:
        return (
            "Ограничение на количество шагов снято. Ты можешь вызывать инструменты "
            "столько раз, сколько потребуется для качественного ответа, затем вызови `final_answer`."
        )
    return (
        f"ВАЖНО: У тебя есть максимум {max_steps} шагов (вызовов инструментов). "
        f"Распределяй шаги экономно и обязательно заверши работу вызовом `final_answer` до исчерпания лимита."
    )


__all__ = [
    "ChunkingConfig",
    "ContextBuilder",
    "ContextItem",
    "TextChunker",
    "TokenChunker",
    "build_step_limit_instruction",
    "compress_history_messages",
]
