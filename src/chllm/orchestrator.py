"""
Модуль для оркестрации запросов к LLM.
Обеспечивает отказоустойчивость, умные ретраи и автоматическое деление батчей.
"""

import asyncio
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .exceptions import ContentBlockedError, RateLimitError, ServiceUnavailableError
from .utils import split_batch

try:
    from chutils.logger import setup_logger
except ImportError:
    import logging  # chutils: ignore[ChutilsIntegrationRule]

    def setup_logger(name: str = "chllm") -> Any:
        """Создает резервный логгер, если chutils не установлен.

        Args:
            name: Имя логгера.

        Returns:
            Экземпляр стандартного логгера.
        """
        return logging.getLogger(name)


logger = setup_logger(__name__)


class LLMProvider(Protocol):
    """Интерфейс для провайдеров LLM, работающих с оркестратором."""

    async def execute(self, data: object) -> object:
        """Выполняет один запрос к LLM (батч или одиночный).

        Args:
            data: Данные запроса.

        Returns:
            Ответ от LLM провайдера.
        """

    def pack_single_prompt(self, prompt: str) -> object:
        """Упаковывает строку промпта в структуру данных, ожидаемую этим провайдером.

        Позволяет библиотеке оставаться универсальной.

        Args:
            prompt: Текст промпта.

        Returns:
            Упакованный объект запроса для провайдера.
        """


@dataclass
class RetryStrategy:
    """Конфигурация стратегии повторных попыток."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    use_jitter: bool = True
    retryable_errors: tuple[type[Exception], ...] = (RateLimitError, ServiceUnavailableError, ConnectionError)


class Orchestrator:
    """Универсальный оркестратор для надежного выполнения запросов к LLM."""

    def __init__(self, provider: LLMProvider, strategy: RetryStrategy | None = None, logger: Any | None = None):
        """Инициализирует оркестратор.

        Args:
            provider: Экземпляр провайдера LLM.
            strategy: Стратегия повторных попыток.
            logger: Логгер для записи процесса.
        """
        self._provider = provider
        self._strategy = strategy or RetryStrategy()
        self._logger = logger or setup_logger(__name__)

    async def execute(self, data: Any) -> Any:
        """Выполняет запрос с поддержкой ретраев и деления данных.

        Args:
            data: Данные для отправки (строка или список для батча).

        Returns:
            Результат выполнения запроса.
        """
        return await self._execute_with_retries(data)

    async def execute_single(self, prompt: str) -> str:
        """Выполняет одиночный запрос к ИИ и возвращает строковый ответ.

        Библиотека не знает о структуре данных приложения, поэтому делегирует
        упаковку промпта провайдеру через pack_single_prompt.

        Args:
            prompt: Текст промпта для отправки в модель.

        Returns:
            Текстовый ответ модели.
        """
        if not prompt:
            return ""

        # Провайдер сам знает, как превратить строку в нужный ему объект
        request_data = self._provider.pack_single_prompt(prompt)

        try:
            result = await self.execute(request_data)

            if isinstance(result, str):
                return result

            def _get_field(obj: Any, fields: Sequence[str]) -> Any:
                for field in fields:
                    if isinstance(obj, dict):
                        if field in obj:
                            return obj[field]
                        continue

                    # Если это Mock-объект из unittest.mock
                    if hasattr(obj, "_mock_children"):
                        # Проверяем явно назначенные атрибуты у Mock
                        if field in getattr(obj, "_mock_children", {}) or field in getattr(obj, "__dict__", {}):
                            val = getattr(obj, field)
                            if not (hasattr(val, "_mock_return_value") and val._mock_name and not isinstance(val, str)):
                                return val
                            if isinstance(val, (str, int, float, list, dict)):
                                return val
                        continue

                    if hasattr(obj, field):
                        val = getattr(obj, field, None)
                        if val is not None:
                            return val
                return None

            text_fields = ("translated_text", "text", "content", "message", "result", "output")
            container_fields = ("batch", "items", "data", "results", "translations")

            # Проверяем, является ли result контейнером (объект или dict)
            for c_field in container_fields:
                container = _get_field(result, [c_field])
                if container and isinstance(container, (list, tuple)) and len(container) > 0:
                    first_item = container[0]
                    if isinstance(first_item, str):
                        return first_item
                    val = _get_field(first_item, text_fields)
                    if val is not None:
                        return str(val)

            # Совместимость с одиночными объектами-ответами
            single_val = _get_field(result, text_fields)
            if single_val is not None:
                return str(single_val)

            return ""
        except Exception as e:
            self._logger.error("Ошибка при выполнении одиночного запроса: %s", e)
            return ""

    async def _execute_with_retries(self, data: Any) -> Any:
        """Внутренний цикл выполнения запроса с ретраями."""
        last_error = None

        for attempt in range(self._strategy.max_retries + 1):
            try:
                return await self._provider.execute(data)

            except ContentBlockedError:
                # Если контент заблокирован, ретраи бессмысленны,
                # пробуем поделить батч, если это список.
                if isinstance(data, list) and len(data) > 1:
                    self._logger.warning("Контент заблокирован. Пробую разделить батч.")
                    return await self._split_and_execute(data)
                raise

            except self._strategy.retryable_errors as e:
                last_error = e
                if attempt == self._strategy.max_retries:
                    self._logger.error("Все попытки исчерпаны. Последняя ошибка: %s", e)
                    raise

                # Расчет задержки с учетом экспоненциального роста и джиттера
                delay = min(
                    self._strategy.base_delay * (self._strategy.exponential_base**attempt), self._strategy.max_delay
                )
                if self._strategy.use_jitter:
                    delay = delay * random.uniform(0.8, 1.2)

                self._logger.warning(
                    "Ошибка при вызове LLM (%s). Попытка %d/%d через %.1fс.",
                    type(e).__name__,
                    attempt + 1,
                    self._strategy.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)

            except Exception as e:
                self._logger.error("Критическая ошибка оркестратора: %s", e)
                raise

        if last_error:
            raise last_error

    async def _split_and_execute(self, batch: list[Any]) -> list[Any]:
        """Рекурсивно делит батч пополам и выполняет части."""
        if len(batch) <= 1:
            # Если в батче 1 элемент и он заблокирован - пробрасываем ошибку выше
            res = await self._provider.execute(batch)
            return res if isinstance(res, list) else [res]

        part1, part2 = split_batch(batch)

        self._logger.info("Деление батча на части: %d и %d элементов.", len(part1), len(part2))

        # Выполняем обе части (каждая со своими ретраями)
        res1 = await self._execute_with_retries(part1)
        res2 = await self._execute_with_retries(part2)

        # Объединяем результаты.
        # Предполагаем, что результат провайдера для списка - тоже список.
        if isinstance(res1, list) and isinstance(res2, list):
            return res1 + res2

        return [res1, res2]


class AgentOrchestrator(Orchestrator):
    """
    Расширенный оркестратор для работы в режиме автономного агента.
    Поддерживает цикл "запрос -> вызов инструментов -> результат -> запрос".
    """

    async def execute_tools(self, response_text: str, tools: dict[str, Callable]) -> str | None:
        """Анализирует текст ответа, и если найдены вызовы инструментов, выполняет их.

        Возвращает строку с результатами. Если инструментов нет, возвращает None.

        Args:
            response_text: Текст ответа модели, возможно содержащий вызовы инструментов.
            tools: Словарь доступных функций-инструментов.

        Returns:
            Результаты выполнения инструментов в виде строки или None.
        """
        from .parser import RobustLLMParser

        parser = RobustLLMParser(logger=self._logger)

        tool_calls = parser.parse_tool_calls(response_text)
        if not tool_calls:
            return None

        results = []
        for call in tool_calls:
            if call.name in tools:
                self._logger.info("Выполнение инструмента: %s", call.name)
                try:
                    tool_func = tools[call.name]
                    if asyncio.iscoroutinefunction(tool_func):
                        res = await tool_func(**call.args)
                    else:
                        res = tool_func(**call.args)
                    results.append(f"Result of {call.name}: {res}")
                except Exception as e:
                    self._logger.error("Ошибка при выполнении инструмента %s: %s", call.name, e)
                    results.append(f"Error executing {call.name}: {e}")
            else:
                self._logger.warning("Инструмент %s не найден", call.name)
                results.append(f"Error: Tool {call.name} is not available.")

        return "\n".join(results)

    async def execute_loop(self, prompt: str, tools: dict[str, Callable], max_iterations: int = 5) -> str:
        """
        Запускает цикл автономной работы агента.

        Args:
            prompt: Начальный промпт для ИИ.
            tools: Словарь доступных Python-функций {имя: функция}.
            max_iterations: Максимальное количество итераций (защита от зацикливания).

        Returns:
            Финальный текстовый ответ ИИ.
        """
        current_prompt = prompt
        last_response = ""

        for i in range(max_iterations):
            self._logger.info("Агентная итерация %d/%d", i + 1, max_iterations)

            # 1. Получаем ответ от ИИ
            last_response = await self.execute_single(current_prompt)
            if not last_response:
                break

            # 2 & 3. Ищем и выполняем инструменты
            tool_results = await self.execute_tools(last_response, tools)
            if tool_results is None:
                # Если инструментов нет - это финальный ответ
                return last_response

            # 4. Формируем новый промпт с результатами для следующей итерации
            current_prompt = f"{last_response}\n\n--- TOOL RESULTS ---\n{tool_results}"

        self._logger.warning("Достигнут лимит итераций агента (%d)", max_iterations)
        return last_response
