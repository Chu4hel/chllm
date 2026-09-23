"""
Модуль для асинхронной пакетной обработки данных с контролем параллелизма.
Позволяет обрабатывать большие списки элементов через LLM с ограничением одновременных запросов.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

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

T_Input = TypeVar("T_Input")
T_Output = TypeVar("T_Output")


@dataclass
class BatchItemResult(Generic[T_Input, T_Output]):
    """Результат обработки одного элемента в пакете.

    Attributes:
        index: Порядковый номер элемента в исходном списке.
        input_item: Исходный элемент.
        output: Результат выполнения (если успешно).
        error: Исключение (если произошла ошибка).
        success: Флаг успешного выполнения.
    """

    index: int
    input_item: T_Input
    output: T_Output | None = None
    error: Exception | None = None

    @property
    def success(self) -> bool:
        """Успешно ли выполнен элемент."""
        return self.error is None


class AsyncBatchProcessor(Generic[T_Input, T_Output]):
    """Асинхронный процессор для параллельной обработки элементов с контролем нагрузки."""

    def __init__(
        self,
        concurrency_limit: int = 5,
        delay_between_requests: float = 0.0,
        logger: Any | None = None,
    ):
        """Инициализирует процессор пакетов.

        Args:
            concurrency_limit: Максимальное количество параллельных задач (семафор).
            delay_between_requests: Дополнительная пауза (в секундах) между запусками задач.
            logger: Опциональный логгер.
        """
        self._concurrency_limit = max(1, concurrency_limit)
        self._delay = delay_between_requests
        self._logger = logger or setup_logger(__name__)

    async def process(
        self,
        items: Sequence[T_Input],
        process_fn: Callable[[T_Input], Awaitable[T_Output]],
        *,
        return_exceptions: bool = True,
    ) -> list[BatchItemResult[T_Input, T_Output]]:
        """Обрабатывает последовательность элементов асинхронно с ограничением параллелизма.

        Args:
            items: Список элементов для обработки.
            process_fn: Асинхронная функция обработки одного элемента.
            return_exceptions: Если True, ошибки сохраняются в поле error результата.
                               Если False, первая же ошибка прерывает весь процесс.

        Returns:
            Список результатов BatchItemResult в том же порядке, что и входные элементы.
        """
        if not items:
            return []

        semaphore = asyncio.Semaphore(self._concurrency_limit)

        async def _worker(index: int, item: T_Input) -> BatchItemResult[T_Input, T_Output]:
            async with semaphore:
                if self._delay > 0:
                    await asyncio.sleep(self._delay)

                try:
                    result = await process_fn(item)
                    return BatchItemResult(index=index, input_item=item, output=result)
                except Exception as e:
                    self._logger.warning("Ошибка обработки элемента #%d: %s", index, e)
                    if not return_exceptions:
                        raise
                    return BatchItemResult(index=index, input_item=item, error=e)

        tasks = [_worker(i, item) for i, item in enumerate(items)]
        results = await asyncio.gather(*tasks, return_exceptions=not return_exceptions)
        return results  # type: ignore[return-value]
