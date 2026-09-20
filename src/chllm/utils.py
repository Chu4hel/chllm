"""Вспомогательные утилиты для работы с батчами и данными."""

from typing import TypeVar

T = TypeVar("T")


def split_batch(batch: list[T]) -> tuple[list[T], list[T]]:
    """Делит батч пополам.

    Если элементов 1 или меньше, первая часть содержит все элементы, вторая пуста.
    Если количество элементов нечетное, первая часть будет на 1 элемент меньше второй (как в orchestrator.py).

    Args:
        batch: Исходный список элементов для разделения.

    Returns:
        Кортеж из двух частей списка.
    """
    if len(batch) <= 1:
        return batch, []

    mid = len(batch) // 2
    return batch[:mid], batch[mid:]
