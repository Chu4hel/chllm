"""Исключения, связанные с вызовами и квотами LLM-провайдеров."""

from .base import CHLLMError


class RateLimitError(CHLLMError):
    """Превышен лимит запросов (RPM/TPM).

    Обычно это временная ошибка, которую можно повторить через паузу.
    """

    def __init__(self, message: str, retry_delay: float = 60.0) -> None:
        super().__init__(message)
        self.retry_delay: float = retry_delay
        "Задержка в секундах перед следующей попыткой"


class QuotaExceededError(CHLLMError):
    """Исчерпана квота (обычно дневной лимит или баланс аккаунта).

    Требует вмешательства пользователя или смены ключа.
    """


class ServiceUnavailableError(CHLLMError):
    """Сервер LLM временно недоступен или перегружен (Error 503/504)."""


class AuthenticationError(CHLLMError):
    """Ошибка аутентификации (неверный API ключ)."""


class ContentBlockedError(CHLLMError):
    """Запрос или ответ заблокирован фильтрами безопасности провайдера."""
