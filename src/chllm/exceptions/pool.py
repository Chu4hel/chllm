"""Исключения, связанные с пулом ключей и ротацией провайдеров."""

from .provider import QuotaExceededError


class NoActiveKeysError(QuotaExceededError):
    """Все API-ключи в пуле исчерпаны, находятся в кулдауне или заблокированы."""
