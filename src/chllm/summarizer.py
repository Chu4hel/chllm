"""Модуль для иерархической обработки и Map-Reduce суммаризации длинных текстов.

Предоставляет универсальные пайплайны HierarchicalProcessor и ChainedSummarizer
для параллельного или последовательного сжатия длинных текстов с гарантированным
соблюдением контекстных лимитов моделей через многоуровневую рекурсивную агрегацию.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from .batch_processor import AsyncBatchProcessor
from .chunker import ChunkingConfig, TextChunker
from .exceptions import CHLLMError
from .orchestrator import LLMProvider, Orchestrator

try:
    from chutils.logger import setup_logger
except ImportError:
    import logging  # chutils: ignore[ChutilsIntegrationRule]

    def setup_logger(name: str = "chllm") -> Any:
        """Создает резервный логгер, если chutils не установлен.

        Args:
            name: Имя логгера.

        Returns:
            Экземпляр логгера.
        """
        return logging.getLogger(name)


logger = setup_logger(__name__)


class MapReduceConfig(BaseModel):
    """Конфигурация для иерархической обработки и Map-Reduce суммаризации."""

    chunk_size: int = Field(
        default=3000,
        ge=10,
        description="Максимальный размер чанка в токенах для этапа Map и промежуточных Reduce-слоев.",
    )
    overlap_tokens: int = Field(
        default=100,
        ge=0,
        description="Количество перекрывающихся токенов между соседними чанками на этапе Map.",
    )
    concurrency_limit: int = Field(
        default=5,
        ge=1,
        description=(
            "Максимальное количество параллельных запросов на этапе Map. При 1 выполняется строго последовательно."
        ),
    )
    delay_between_requests: float = Field(
        default=0.0,
        ge=0.0,
        description="Задержка в секундах между запусками задач на этапе Map.",
    )
    max_reduce_levels: int = Field(
        default=5,
        ge=1,
        description="Максимальное количество итераций рекурсивного сжатия на этапе Reduce.",
    )
    map_prompt_template: str = Field(
        default=(
            "Сделай краткую, но информативную выжимку следующего фрагмента текста, "
            "сохраняя ключевые факты, события и имена:\n\n{text}"
        ),
        description="Шаблон промпта для обработки каждого чанка (этап Map).",
    )
    reduce_prompt_template: str = Field(
        default=(
            "Ниже приведены промежуточные выжимки частей длинного текста.\n"
            "Объедини их в единый связный, структурированный и непротиворечивый итоговый текст:\n\n{text}"
        ),
        description="Шаблон промпта для слияния промежуточных выжимок (этап Reduce).",
    )
    single_prompt_template: str | None = Field(
        default=None,
        description=(
            "Опциональный шаблон промпта, если текст умещается в один чанк. "
            "Если None, используется map_prompt_template."
        ),
    )
    refine_prompt_template: str = Field(
        default=(
            "Текущая выжимка предшествующего контекста:\n{previous_summary}\n\n"
            "Новый фрагмент текста:\n{text}\n\n"
            "Обнови и дополни выжимку с учетом новой информации, сохраняя связность и структуру."
        ),
        description="Шаблон промпта для последовательного режима уточнения (Refine).",
    )
    summary_prefix: str = Field(
        default="Часть {index}:\n{summary}",
        description="Шаблон форматирования отдельной локальной выжимки перед объединением в Reduce.",
    )
    fail_on_chunk_error: bool = Field(
        default=True,
        description="Прерывать ли процесс при ошибке обработки отдельного чанка.",
    )


class HierarchicalProcessor:
    """Универсальный процессор для сжатия и суммаризации длинных текстов по паттерну Map-Reduce."""

    def __init__(
        self,
        orchestrator: Orchestrator | LLMProvider,
        config: MapReduceConfig | None = None,
        chunker: TextChunker | None = None,
        logger: Any | None = None,
    ) -> None:
        """Инициализирует процессор.

        Args:
            orchestrator: Экземпляр Orchestrator или совместимый LLMProvider.
            config: Опциональная конфигурация MapReduceConfig.
            chunker: Опциональный кастомный TextChunker.
            logger: Опциональный логгер.
        """
        self._orchestrator = orchestrator if isinstance(orchestrator, Orchestrator) else Orchestrator(orchestrator)
        self.config = config or MapReduceConfig()
        self._chunker = chunker or TextChunker(
            config=ChunkingConfig(
                max_tokens=self.config.chunk_size,
                overlap_tokens=self.config.overlap_tokens,
            )
        )
        self._logger = logger or setup_logger(__name__)

    def _format_prompt(
        self,
        template: str,
        text: str,
        *,
        index: int = 1,
        total: int = 1,
        previous_summary: str = "",
    ) -> str:
        """Форматирует строку промпта безопасной подстановкой плейсхолдеров.

        Args:
            template: Шаблон промпта.
            text: Основной текст чанка или объединенных выжимок.
            index: Порядковый номер части (с 1).
            total: Общее количество частей.
            previous_summary: Выжимка предшествующего контекста (для режима refine).

        Returns:
            Готовый текст промпта.
        """
        result = template
        for placeholder in ("{{text}}", "{text}", "{{chunk}}", "{chunk}"):
            result = result.replace(placeholder, text)
        for placeholder in ("{{index}}", "{index}"):
            result = result.replace(placeholder, str(index))
        for placeholder in ("{{total}}", "{total}"):
            result = result.replace(placeholder, str(total))
        for placeholder in ("{{previous_summary}}", "{previous_summary}"):
            result = result.replace(placeholder, previous_summary)
        return result

    def _format_summary_block(self, index: int, summary: str) -> str:
        """Форматирует блок выжимки с указанием индекса части.

        Args:
            index: Порядковый номер.
            summary: Текст выжимки.

        Returns:
            Отформатированная строка блока.
        """
        prefix = self.config.summary_prefix
        for ph in ("{{index}}", "{index}"):
            prefix = prefix.replace(ph, str(index))
        for ph in ("{{summary}}", "{summary}", "{{text}}", "{text}"):
            prefix = prefix.replace(ph, summary)
        return prefix

    async def _notify_progress(
        self,
        callback: Callable[[str, int, int], Awaitable[None] | None] | None,
        phase: str,
        current: int,
        total: int,
    ) -> None:
        """Безопасно вызывает коллбэк прогресса, поддерживая синхронные и асинхронные функции.

        Args:
            callback: Функция обратного вызова прогресса.
            phase: Текущая фаза ('map', 'reduce', 'refine', 'single').
            current: Номер текущего шага.
            total: Общее число шагов.
        """
        if callback is None:
            return
        try:
            res = callback(phase, current, total)
            if inspect.isawaitable(res):
                await res
        except Exception as e:
            self._logger.warning("Ошибка в обработчике прогресса: %s", e)

    async def process_text(
        self,
        text: str,
        *,
        on_progress: Callable[[str, int, int], Awaitable[None] | None] | None = None,
    ) -> str:
        """Разбивает текст на чанки через TextChunker и выполняет Map-Reduce обработку.

        Args:
            text: Исходный длинный текст.
            on_progress: Опциональный коллбэк отслеживания прогресса.

        Returns:
            Итоговый сжатый/суммаризированный текст.
        """
        clean_text = text.strip()
        if not clean_text:
            return ""

        chunks = self._chunker.split_text(clean_text)
        return await self.process_chunks(chunks, on_progress=on_progress)

    async def process_chunks(
        self,
        chunks: Sequence[str],
        *,
        on_progress: Callable[[str, int, int], Awaitable[None] | None] | None = None,
    ) -> str:
        """Обрабатывает список готовых текстовых фрагментов по алгоритму Map-Reduce.

        Args:
            chunks: Список фрагментов текста для обработки.
            on_progress: Опциональный коллбэк для отслеживания прогресса (фаза, текущий, всего).

        Returns:
            Финальный сжатый/суммаризированный текст.

        Raises:
            CHLLMError: Если произошла ошибка при обработке чанка (при fail_on_chunk_error=True).
        """
        if not chunks:
            return ""

        total_chunks = len(chunks)

        # 1. Оптимизация для одиночного чанка
        if total_chunks == 1:
            await self._notify_progress(on_progress, "single", 1, 1)
            single_template = self.config.single_prompt_template or self.config.map_prompt_template
            prompt = self._format_prompt(single_template, chunks[0], index=1, total=1)
            result = await self._orchestrator.execute_single(prompt)
            return result or ""

        # 2. Фаза Map: параллельная или последовательная обработка чанков
        self._logger.info("Старт фазы Map: %d чанков (concurrency=%d)", total_chunks, self.config.concurrency_limit)
        map_results: list[str] = []

        if self.config.concurrency_limit == 1:
            for idx, chunk in enumerate(chunks):
                prompt = self._format_prompt(self.config.map_prompt_template, chunk, index=idx + 1, total=total_chunks)
                try:
                    summary = await self._orchestrator.execute_single(prompt)
                    map_results.append(summary or "")
                except Exception as e:
                    self._logger.error("Ошибка при обработке чанка %d/%d: %s", idx + 1, total_chunks, e)
                    if self.config.fail_on_chunk_error:
                        raise CHLLMError(f"Ошибка обработки чанка {idx + 1}: {e}") from e
                await self._notify_progress(on_progress, "map", idx + 1, total_chunks)
        else:
            processor = AsyncBatchProcessor(
                concurrency_limit=self.config.concurrency_limit,
                delay_between_requests=self.config.delay_between_requests,
                logger=self._logger,
            )

            completed_count = 0

            async def _map_worker(item: tuple[int, str]) -> str:
                nonlocal completed_count
                idx, chunk_text = item
                prompt = self._format_prompt(
                    self.config.map_prompt_template,
                    chunk_text,
                    index=idx + 1,
                    total=total_chunks,
                )
                res = await self._orchestrator.execute_single(prompt)
                completed_count += 1
                await self._notify_progress(on_progress, "map", completed_count, total_chunks)
                return res or ""

            batch_results = await processor.process(
                list(enumerate(chunks)),
                _map_worker,
                return_exceptions=True,
            )

            for item_res in batch_results:
                if not item_res.success:
                    self._logger.error(
                        "Ошибка при параллельной обработке чанка %d: %s",
                        item_res.index + 1,
                        item_res.error,
                    )
                    if self.config.fail_on_chunk_error:
                        raise CHLLMError(
                            f"Ошибка обработки чанка {item_res.index + 1}: {item_res.error}"
                        ) from item_res.error
                map_results.append(item_res.output or "")

        # Отфильтруем пустые результаты, если таковые появились
        intermediate_summaries = [s for s in map_results if s.strip()]
        if not intermediate_summaries:
            return ""

        if len(intermediate_summaries) == 1:
            return intermediate_summaries[0]

        # 3. Фаза Reduce: иерархическая рекурсивная агрегация
        return await self._hierarchical_reduce(intermediate_summaries, on_progress=on_progress)

    async def _hierarchical_reduce(
        self,
        summaries: list[str],
        *,
        on_progress: Callable[[str, int, int], Awaitable[None] | None] | None = None,
    ) -> str:
        """Выполняет иерархическое слияние промежуточных выжимок с контролем лимита токенов.

        Args:
            summaries: Список промежуточных текстовых выжимок.
            on_progress: Опциональный коллбэк прогресса.

        Returns:
            Итоговый объединенный текст.

        Raises:
            CHLLMError: Если не удалось выполнить слияние в пределах максимального числа уровней.
        """
        current_items = list(summaries)
        level = 0

        while len(current_items) > 1 and level < self.config.max_reduce_levels:
            level += 1
            formatted_blocks = [self._format_summary_block(idx + 1, text) for idx, text in enumerate(current_items)]
            merged_candidate = "\n\n".join(formatted_blocks)

            # Проверяем, помещаются ли все выжимки текущего уровня в безопасный лимит одного Reduce-промпта
            tokens_count = self._chunker.count_tokens(merged_candidate)
            if tokens_count <= self.config.chunk_size:
                self._logger.info(
                    "Reduce уровень %d: объединение %d выжимок в финал (%d токенов)",
                    level,
                    len(current_items),
                    tokens_count,
                )
                prompt = self._format_prompt(
                    self.config.reduce_prompt_template,
                    merged_candidate,
                    index=1,
                    total=1,
                )
                await self._notify_progress(on_progress, "reduce", level, level)
                final_res = await self._orchestrator.execute_single(prompt)
                return final_res or ""

            # Если объем слишком велик для одного промпта, нарезаем промежуточные выжимки на батчи
            self._logger.info(
                "Reduce уровень %d: объем выжимок (%d токенов) превышает chunk_size (%d), запуск иерархического сжатия",
                level,
                tokens_count,
                self.config.chunk_size,
            )
            batches = self._chunker.split_lines(formatted_blocks, max_tokens=self.config.chunk_size)
            if len(batches) >= len(current_items):
                # Невозможно больше сгруппировать блоки по строкам, используем split_text
                batches = self._chunker.split_text(merged_candidate)

            next_level_summaries: list[str] = []
            for b_idx, batch_text in enumerate(batches):
                reduce_prompt = self._format_prompt(
                    self.config.reduce_prompt_template,
                    batch_text,
                    index=b_idx + 1,
                    total=len(batches),
                )
                await self._notify_progress(on_progress, f"reduce_l{level}", b_idx + 1, len(batches))
                sub_summary = await self._orchestrator.execute_single(reduce_prompt)
                if sub_summary and sub_summary.strip():
                    next_level_summaries.append(sub_summary)

            if not next_level_summaries:
                break

            current_items = next_level_summaries

        if len(current_items) == 1:
            return current_items[0]

        # Финальный fallback, если достигли max_reduce_levels
        final_prompt = self._format_prompt(
            self.config.reduce_prompt_template,
            "\n\n".join(current_items),
            index=1,
            total=1,
        )
        final_result = await self._orchestrator.execute_single(final_prompt)
        return final_result or ""


class ChainedSummarizer(HierarchicalProcessor):
    """Специализированный суммаризатор с поддержкой Map-Reduce и последовательного Refine-режима."""

    def __init__(
        self,
        orchestrator: Orchestrator | LLMProvider,
        config: MapReduceConfig | None = None,
        chunker: TextChunker | None = None,
        logger: Any | None = None,
        chain_mode: str = "map_reduce",
    ) -> None:
        """Инициализирует суммаризатор.

        Args:
            orchestrator: Экземпляр Orchestrator или LLMProvider.
            config: Конфигурация Map-Reduce.
            chunker: Опциональный кастомный TextChunker.
            logger: Опциональный логгер.
            chain_mode: Режим суммаризации: 'map_reduce' (по умолчанию) или 'refine' ('sequential').
        """
        super().__init__(orchestrator=orchestrator, config=config, chunker=chunker, logger=logger)
        self.chain_mode = chain_mode.lower()

    async def summarize(
        self,
        text: str,
        *,
        on_progress: Callable[[str, int, int], Awaitable[None] | None] | None = None,
    ) -> str:
        """Выполняет суммаризацию переданного длинного текста.

        Args:
            text: Исходный текст.
            on_progress: Опциональный коллбэк отслеживания прогресса.

        Returns:
            Итоговая выжимка текста.
        """
        clean_text = text.strip()
        if not clean_text:
            return ""

        chunks = self._chunker.split_text(clean_text)
        return await self.summarize_chunks(chunks, on_progress=on_progress)

    async def summarize_chunks(
        self,
        chunks: Sequence[str],
        *,
        on_progress: Callable[[str, int, int], Awaitable[None] | None] | None = None,
    ) -> str:
        """Выполняет суммаризацию списка фрагментов текста в соответствии с выбранным chain_mode.

        Args:
            chunks: Список фрагментов текста.
            on_progress: Опциональный коллбэк отслеживания прогресса.

        Returns:
            Итоговая выжимка текста.
        """
        if not chunks:
            return ""

        if self.chain_mode in ("refine", "sequential"):
            return await self._refine_chunks(chunks, on_progress=on_progress)
        return await self.process_chunks(chunks, on_progress=on_progress)

    async def _refine_chunks(
        self,
        chunks: Sequence[str],
        *,
        on_progress: Callable[[str, int, int], Awaitable[None] | None] | None = None,
    ) -> str:
        """Последовательно уточняет выжимку от первого чанка к последнему (Refine chain).

        Args:
            chunks: Список фрагментов текста.
            on_progress: Опциональный коллбэк отслеживания прогресса.

        Returns:
            Финальная накопленная и уточненная выжимка.

        Raises:
            CHLLMError: Если произошла ошибка при обработке чанка (при fail_on_chunk_error=True).
        """
        total = len(chunks)
        if total == 1:
            single_template = self.config.single_prompt_template or self.config.map_prompt_template
            prompt = self._format_prompt(single_template, chunks[0], index=1, total=1)
            await self._notify_progress(on_progress, "refine", 1, 1)
            result = await self._orchestrator.execute_single(prompt)
            return result or ""

        # Шаг 1: Начальная выжимка первого чанка
        first_prompt = self._format_prompt(self.config.map_prompt_template, chunks[0], index=1, total=total)
        current_summary = await self._orchestrator.execute_single(first_prompt) or ""
        await self._notify_progress(on_progress, "refine", 1, total)

        # Шаг 2..N: Последовательное уточнение
        for idx in range(1, total):
            chunk = chunks[idx]
            refine_prompt = self._format_prompt(
                self.config.refine_prompt_template,
                chunk,
                index=idx + 1,
                total=total,
                previous_summary=current_summary,
            )
            try:
                updated_summary = await self._orchestrator.execute_single(refine_prompt)
                if updated_summary and updated_summary.strip():
                    current_summary = updated_summary
            except Exception as e:
                self._logger.error("Ошибка при уточняющей суммаризации чанка %d/%d: %s", idx + 1, total, e)
                if self.config.fail_on_chunk_error:
                    raise CHLLMError(f"Ошибка в этапе refine на чанке {idx + 1}: {e}") from e
            await self._notify_progress(on_progress, "refine", idx + 1, total)

        return current_summary
