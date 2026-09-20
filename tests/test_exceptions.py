from chllm.exceptions import (
    CHLLMError,
    QuotaExceededError,
    RateLimitError,
    ServiceUnavailableError,
)


def test_exception_hierarchy():
    # Проверяем, что все ошибки наследуются от базового класса библиотеки
    assert issubclass(RateLimitError, CHLLMError)
    assert issubclass(QuotaExceededError, CHLLMError)
    assert issubclass(ServiceUnavailableError, CHLLMError)


def test_rate_limit_error_has_retry_delay():
    # Ошибка лимита должна поддерживать передачу времени ожидания
    error = RateLimitError("Too many requests", retry_delay=45.0)
    assert error.retry_delay == 45.0
    assert "Too many requests" in str(error)


def test_default_retry_delay():
    error = RateLimitError("Limit")
    assert error.retry_delay == 60.0  # Значение по умолчанию
