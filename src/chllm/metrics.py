"""
Модуль для работы с метриками и подсчета токенов в запросах к LLM.
"""

import json
import math
from typing import Any

from pydantic import BaseModel, Field, computed_field


class UsageMetrics(BaseModel):
    """Модель для хранения данных об использовании токенов.

    Attributes:
        prompt_tokens: Количество токенов в запросе.
        candidates_tokens: Количество токенов в ответе.
    """

    prompt_tokens: int = Field(default=0, ge=0)
    candidates_tokens: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        """Вычисляет общую сумму токенов (вычисляемое поле)."""
        return self.prompt_tokens + self.candidates_tokens


class TokenCounter:
    """Универсальный калькулятор для оценки количества токенов.

    Поддерживает различные стратегии подсчета: от простой эвристики
    до использования библиотек-токенизаторов.
    """

    def __init__(
        self, strategy: str = "heuristic", model: str | None = None, chars_per_token: dict[str, float] | None = None
    ):
        """Инициализирует счетчик.

        Args:
            strategy: Стратегия подсчета ("heuristic", "tiktoken").
            model: Имя модели для точного подбора токенизатора.
            chars_per_token: Кастомные коэффициенты символов на токен.
        """
        self._strategy = strategy
        self._model = model

        # Константы по умолчанию (символов на токен)
        default_ratios = {"en": 2.5, "ru": 2.5, "default": 3.5}

        # Объединяем дефолты с пользовательскими настройками
        self._chars_per_token = default_ratios
        if chars_per_token:
            self._chars_per_token.update(chars_per_token)

    def count(self, data: Any, language: str | None = None) -> int:
        """Подсчитывает или оценивает количество токенов в данных.

        Args:
            data: Текст, словарь или список данных для подсчета.
            language: Код языка. Если None, используется коэффициент 'default'.

        Returns:
            Оценочное или точное количество токенов.
        """
        if data is None:
            return 0

        # Превращаем данные в строку для оценки
        text = self._to_string(data)

        if self._strategy == "heuristic":
            # Если язык не указан, используем 'default'
            return self._heuristic_count(text, language or "default")

        # TODO: Реализовать интеграцию с tiktoken при необходимости
        return self._heuristic_count(text, language or "default")

    def estimate_completion_tokens(
        self, items: list[str], scaling_factor: float = 1.2, per_item_overhead_chars: int = 50, language: str = "ru"
    ) -> int:
        """Оценивает примерное количество токенов в будущем ответе (Completion) ИИ.

        Полезно для планирования лимитов и логирования ожидаемой нагрузки.

        Args:
            items: Список исходных строк (элементов), которые будут обработаны.
            scaling_factor: Коэффициент изменения объема текста (1.2 для перевода, 0.5 для суммаризации).
            per_item_overhead_chars: Технический оверхед на один элемент (структура JSON).
            language: Ожидаемый язык ответа.

        Returns:
            Приблизительное количество токенов в ответе.
        """
        if not items:
            return 0

        total_chars: float = 0.0
        for text in items:
            # Считаем длину результата + оверхед
            item_len = len(text) * scaling_factor
            total_chars += item_len + per_item_overhead_chars

        return self.count(" " * int(total_chars), language=language)

    def _heuristic_count(self, text: str, language: str) -> int:
        """Оценивает количество токенов по количеству символов."""
        chars_per_token = self._chars_per_token.get(language, self._chars_per_token["default"])
        if chars_per_token <= 0:
            return 0
        return math.ceil(len(text) / chars_per_token)

    @staticmethod
    def _to_string(data: Any) -> str:
        """Приводит любые входные данные к строковому виду."""
        if isinstance(data, str):
            return data
        if isinstance(data, (dict, list)):
            return json.dumps(data, ensure_ascii=False)
        return str(data)
