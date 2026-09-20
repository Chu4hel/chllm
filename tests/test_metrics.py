import pytest

from chllm.metrics import TokenCounter, UsageMetrics


def test_usage_metrics_initialization():
    metrics = UsageMetrics(prompt_tokens=10, candidates_tokens=5)
    assert metrics.prompt_tokens == 10
    assert metrics.candidates_tokens == 5
    assert metrics.total_tokens == 15


def test_heuristic_token_counter_english():
    # Эвристика по умолчанию: 2.5 символа на токен
    counter = TokenCounter(strategy="heuristic")
    text = "Hello world"  # 11 символов
    # 11 / 2.5 = 4.4 -> 5 (округление вверх)
    assert counter.count(text, language="en") == 5


def test_heuristic_token_counter_russian():
    # Эвристика по умолчанию: 2.5 символа на токен
    counter = TokenCounter(strategy="heuristic")
    text = "Привет мир"  # 10 символов
    # 10 / 2.5 = 4
    assert counter.count(text, language="ru") == 4


def test_token_counter_with_custom_ratios():
    # Тест ручной настройки коэффициентов
    counter = TokenCounter(chars_per_token={"en": 4.0, "ru": 1.5})

    # English: 11 / 4.0 = 2.75 -> 3
    assert counter.count("Hello world", language="en") == 3

    # Russian: 10 / 1.5 = 6.66 -> 7
    assert counter.count("Привет мир", language="ru") == 7


def test_token_counter_with_dict():
    # Подсчет токенов в словаре (JSON-структуре)
    # JSON: '{"key": "value"}' -> 16 символов
    # По умолчанию используется 3.5 для неизвестных языков/структур
    counter = TokenCounter(strategy="heuristic")
    data = {"key": "value"}
    # 16 / 3.5 = 4.57 -> 5
    assert counter.count(data) == 5


@pytest.mark.skip(reason="Требует tiktoken, проверим позже")
def test_tiktoken_counter():
    counter = TokenCounter(strategy="tiktoken", model="gpt-4")
    assert counter.count("Hello world") > 0
