"""Интеллектуальный чанкер текстов и списков по лимиту токенов."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from pydantic import BaseModel, Field

from .metrics import TokenCounter

T = TypeVar("T")


class ChunkingConfig(BaseModel):
    """Конфигурация параметров нарезки текста на чанки."""

    max_tokens: int = Field(default=2000, gt=0, description="Максимальный размер чанка в токенах.")
    overlap_tokens: int = Field(default=0, ge=0, description="Количество токенов перекрытия между чанками.")
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
        description="Иерархия разделителей для смыслового деления текста.",
    )
    strip_whitespace: bool = Field(default=True, description="Очищать ли пробельные символы по краям чанков.")

    @property
    def chunk_size(self) -> int:
        """Псевдоним для max_tokens."""
        return self.max_tokens


class TextChunker:
    """Универсальный чанкер для разбиения текстов и списков с гарантией лимита токенов."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        token_counter: TokenCounter | Callable[[str], int] | None = None,
    ) -> None:
        """Инициализирует TextChunker.

        Args:
            config: Конфигурация нарезки (лимит токенов, разделители, перекрытие).
            token_counter: Экземпляр TokenCounter или кастомная функция подсчета токенов.
        """
        self.config = config or ChunkingConfig()

        if token_counter is None:
            self._counter_fn: Callable[[str], int] = TokenCounter().count
        elif isinstance(token_counter, TokenCounter):
            self._counter_fn = token_counter.count
        else:
            self._counter_fn = token_counter

    @property
    def token_counter(self) -> Callable[[str], int]:
        """Функция подсчета токенов."""
        return self._counter_fn

    def count_tokens(self, text: str) -> int:
        """Возвращает количество токенов в переданном тексте.

        Args:
            text: Исходная строка.

        Returns:
            Количество токенов.
        """
        if not text:
            return 0
        return self._counter_fn(text)

    def split_text(
        self,
        text: str,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        separators: list[str] | None = None,
    ) -> list[str]:
        """Разбивает сплошной текст на чанки с гарантированным непревышением лимита токенов.

        Использует иерархический поиск разделителей (абзацы -> строки -> предложения -> слова -> символы).

        Args:
            text: Исходный текст для разделения.
            max_tokens: Максимальный размер чанка (по умолчанию из config).
            overlap_tokens: Размер перекрытия чанков (по умолчанию из config).
            separators: Кастомный список разделителей по приоритету.

        Returns:
            Список строковых чанков.
        """
        if not text or not text.strip():
            return []

        limit = max_tokens or self.config.max_tokens
        overlap = overlap_tokens if overlap_tokens is not None else self.config.overlap_tokens
        seps = separators or self.config.separators

        if self.count_tokens(text) <= limit:
            cleaned = text.strip() if self.config.strip_whitespace else text
            return [cleaned] if cleaned else []

        raw_chunks = self._recursive_split(text, limit, seps)

        if overlap <= 0 or len(raw_chunks) <= 1:
            if self.config.strip_whitespace:
                return [c.strip() for c in raw_chunks if c.strip()]
            return [c for c in raw_chunks if c]

        return self._apply_overlap(raw_chunks, limit, overlap)

    def _recursive_split(self, text: str, max_tokens: int, separators: list[str]) -> list[str]:
        """Рекурсивно делит текст по разделителям до достижения заданного размера токенов."""
        if not text:
            return []

        if self.count_tokens(text) <= max_tokens:
            return [text]

        # Ищем подходящий разделитель
        chosen_sep = ""
        new_separators: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                chosen_sep = ""
                new_separators = []
                break
            if sep in text:
                chosen_sep = sep
                new_separators = separators[i + 1 :]
                break

        # Если делим по символам (крайний fallback)
        if chosen_sep == "":
            return self._split_by_characters(text, max_tokens)

        # Разбиваем по выбранному разделителю
        parts = text.split(chosen_sep)
        chunks: list[str] = []
        current_chunk: list[str] = []

        for part in parts:
            if not part:
                continue

            test_parts = [*current_chunk, part]
            test_text = chosen_sep.join(test_parts)

            if self.count_tokens(test_text) <= max_tokens:
                current_chunk.append(part)
            else:
                if current_chunk:
                    chunks.append(chosen_sep.join(current_chunk))
                    current_chunk = []

                # Если одиночная часть все равно превышает лимит, рекурсивно дробим более мелким разделителем
                if self.count_tokens(part) > max_tokens:
                    sub_chunks = self._recursive_split(part, max_tokens, new_separators)
                    for sc in sub_chunks:
                        if chunks and self.count_tokens(chunks[-1] + chosen_sep + sc) <= max_tokens:
                            chunks[-1] = chunks[-1] + chosen_sep + sc
                        else:
                            chunks.append(sc)
                else:
                    current_chunk.append(part)

        if current_chunk:
            chunks.append(chosen_sep.join(current_chunk))

        return chunks

    def _split_by_characters(self, text: str, max_tokens: int) -> list[str]:
        """Посимвольное разбиение для текста без пробелов и пунктуации."""
        chunks: list[str] = []
        current = ""
        for char in text:
            test = current + char
            if self.count_tokens(test) <= max_tokens:
                current = test
            else:
                if current:
                    chunks.append(current)
                current = char
        if current:
            chunks.append(current)
        return chunks

    def _apply_overlap(self, chunks: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
        """Добавляет скользящее окно перекрытия между последовательными чанками."""
        result: list[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk.strip() if self.config.strip_whitespace else chunk)
                continue

            prev_chunk = chunks[i - 1]
            prev_words = prev_chunk.split()
            overlap_prefix_words: list[str] = []

            for w in reversed(prev_words):
                candidate_words = [w, *overlap_prefix_words]
                candidate_text = " ".join(candidate_words)
                if self.count_tokens(candidate_text) <= overlap_tokens:
                    overlap_prefix_words = candidate_words
                else:
                    break

            if overlap_prefix_words:
                prefix = " ".join(overlap_prefix_words)
                combined = f"{prefix}\n{chunk}"
                if self.count_tokens(combined) <= max_tokens:
                    chunk = combined

            cleaned = chunk.strip() if self.config.strip_whitespace else chunk
            if cleaned:
                result.append(cleaned)

        return result

    def split_lines(
        self,
        lines: Sequence[str],
        max_tokens: int | None = None,
        separator: str = "\n",
    ) -> list[str]:
        """Объединяет строки в батчи с непревышением лимита токенов в каждом батче.

        Полезно для нарезки списков диалогов, логов или скриптов для суммаризации и перевода.

        Args:
            lines: Последовательность строк.
            max_tokens: Лимит токенов на один батч.
            separator: Разделитель между строками (по умолчанию '\\n').

        Returns:
            Список сформированных текстовых чанков.
        """
        if not lines:
            return []

        limit = max_tokens or self.config.max_tokens
        batches: list[str] = []
        current_lines: list[str] = []

        for raw_line in lines:
            line = raw_line.strip() if self.config.strip_whitespace else raw_line
            if not line:
                continue

            test_batch = [*current_lines, line]
            test_tokens = self.count_tokens(separator.join(test_batch))

            if test_tokens <= limit:
                current_lines.append(line)
            else:
                if current_lines:
                    batches.append(separator.join(current_lines))
                    current_lines = []

                # Если сама одиночная строка больше лимита, дробим ее через split_text
                if self.count_tokens(line) > limit:
                    sub_chunks = self.split_text(line, max_tokens=limit)
                    batches.extend(sub_chunks[:-1])
                    if sub_chunks:
                        current_lines = [sub_chunks[-1]]
                else:
                    current_lines.append(line)

        if current_lines:
            batches.append(separator.join(current_lines))

        return batches

    def split_items(
        self,
        items: Sequence[T],
        max_tokens: int | None = None,
        item_to_text: Callable[[T], str] = str,
    ) -> list[list[T]]:
        """Разбивает список произвольных объектов на подсписки по суммарному лимиту токенов.

        Args:
            items: Исходный список объектов.
            max_tokens: Лимит токенов на одну группу объектов.
            item_to_text: Функция преобразования объекта в текст для оценки токенов.

        Returns:
            Список списков сгруппированных объектов.
        """
        if not items:
            return []

        limit = max_tokens or self.config.max_tokens
        batches: list[list[T]] = []
        current_batch: list[T] = []
        current_tokens = 0

        for item in items:
            item_tokens = self.count_tokens(item_to_text(item))

            if current_batch and (current_tokens + item_tokens > limit):
                batches.append(current_batch)
                current_batch = [item]
                current_tokens = item_tokens
            else:
                current_batch.append(item)
                current_tokens += item_tokens

        if current_batch:
            batches.append(current_batch)

        return batches


TokenChunker = TextChunker
"""Псевдоним для TextChunker."""
