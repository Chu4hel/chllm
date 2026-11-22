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


def mask_proxy_url(proxy_url: str | None) -> str:
    """Маскирует учетные данные в URL-строке прокси для безопасного логирования.

    Args:
        proxy_url: URL прокси-сервера (например, 'http://user:password@host:port').

    Returns:
        Строка URL с замаскированным паролем или исходная строка.
    """
    if not proxy_url:
        return ""
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(proxy_url)
        if parsed.password:
            new_netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
            return urlunparse(parsed._replace(netloc=new_netloc))
    except Exception:
        pass
    return proxy_url
