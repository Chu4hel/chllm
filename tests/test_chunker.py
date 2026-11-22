"""Тесты для TextChunker, TokenChunker и разбиения текста по лимиту токенов."""

from dataclasses import dataclass

from chllm import ChunkingConfig, TextChunker, TokenChunker, TokenCounter


def test_alias_identity():
    """Проверка, что TokenChunker является псевдонимом TextChunker."""
    assert TokenChunker is TextChunker


def test_empty_input():
    """Проверка обработки пустых строк и списков."""
    chunker = TextChunker()
    assert chunker.split_text("") == []
    assert chunker.split_text("   \n\t  ") == []
    assert chunker.split_lines([]) == []
    assert chunker.split_items([]) == []


def test_small_text_single_chunk():
    """Текст меньше лимита токенов возвращается как один чанк."""
    chunker = TextChunker(ChunkingConfig(max_tokens=100))
    text = "Короткий текст, который легко умещается в лимит."
    chunks = chunker.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_by_paragraphs():
    """Разбиение текста по параграфам с соблюдением лимита."""
    p1 = "Первый параграф текста с некоторым содержанием."
    p2 = "Второй параграф текста с описанием других событий."
    p3 = "Третий параграф текста, завершающий мысль."
    full_text = f"{p1}\n\n{p2}\n\n{p3}"

    # Настраиваем счетчик так, чтобы 1 параграф помещался, а 2 уже нет
    counter = TokenCounter()
    p1_tokens = counter.count(p1)
    limit = int(p1_tokens * 1.5)

    chunker = TextChunker(ChunkingConfig(max_tokens=limit))
    chunks = chunker.split_text(full_text)

    assert len(chunks) == 3
    assert chunks[0] == p1
    assert chunks[1] == p2
    assert chunks[2] == p3
    for chunk in chunks:
        assert chunker.count_tokens(chunk) <= limit


def test_split_by_sentences_and_words():
    """Разбиение сплошного длинного текста без переносов строк по предложениям."""
    s1 = "Это первое важное предложение."
    s2 = "Это второе предложение, развивающее мысль."
    s3 = "А это третье предложение, подводящее итог."
    full_text = f"{s1} {s2} {s3}"

    counter = TokenCounter()
    limit = counter.count(s1) + 2

    chunker = TextChunker(ChunkingConfig(max_tokens=limit))
    chunks = chunker.split_text(full_text)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunker.count_tokens(chunk) <= limit


def test_extreme_unspaced_string_fallback():
    """Разбиение гигантской строки без пробелов (символьный fallback)."""
    # 200 символов подряд без пробелов
    huge_unspaced = "A" * 200
    chunker = TextChunker(ChunkingConfig(max_tokens=20))
    chunks = chunker.split_text(huge_unspaced)

    assert len(chunks) > 1
    # Ни один чанк не превышает лимит токенов
    for chunk in chunks:
        assert chunker.count_tokens(chunk) <= 20
    # Склеенный текст равен оригиналу
    assert "".join(chunks) == huge_unspaced


def test_overlap_sliding_window():
    """Проверка перекрытия между соседними чанками."""
    text = (
        "Раз два три четыре пять. "
        "Шесть семь восемь девять десять. "
        "Одиннадцать двенадцать тринадцать четырнадцать пятнадцать."
    )
    chunker = TextChunker(ChunkingConfig(max_tokens=25, overlap_tokens=8))
    chunks = chunker.split_text(text)

    assert len(chunks) >= 2
    # Проверяем, что во втором чанке есть слова из конца первого
    first_chunk_words = set(chunks[0].split())
    second_chunk_words = set(chunks[1].split())
    assert len(first_chunk_words.intersection(second_chunk_words)) > 0


def test_split_lines_dialogues():
    """Проверка нарезки диалоговых строк на батчи (паттерн StorySummarizationService)."""
    lines = [
        "Eileen: Привет! Как твои дела?",
        "Player: Все отлично, перевожу новеллу.",
        "Eileen: Это замечательно! ИИ справляется?",
        "Player: Да, батчи нарезаются идеально.",
        "Eileen: Тогда продолжим наше приключение.",
    ]
    counter = TokenCounter()
    two_lines_tokens = counter.count("\n".join(lines[:2]))
    limit = two_lines_tokens + 2

    chunker = TextChunker(ChunkingConfig(max_tokens=limit))
    batches = chunker.split_lines(lines)

    assert len(batches) >= 2
    for batch in batches:
        assert chunker.count_tokens(batch) <= limit
        assert "\n" in batch or len(batch.splitlines()) == 1


def test_split_lines_single_oversized_line():
    """Строка, которая сама по себе больше лимита, дробится без ошибок."""
    oversized_line = "Слишком длинная строка диалога " * 20
    lines = ["Короткая строка 1", oversized_line, "Короткая строка 2"]

    chunker = TextChunker(ChunkingConfig(max_tokens=30))
    batches = chunker.split_lines(lines)

    assert len(batches) > 1
    for b in batches:
        assert chunker.count_tokens(b) <= 30


@dataclass
class DummyItem:
    item_id: int
    text: str


def test_split_items():
    """Проверка нарезки произвольных объектов на группы."""
    items = [
        DummyItem(item_id=1, text="Краткое описание номер один"),
        DummyItem(item_id=2, text="Второе описание элемента"),
        DummyItem(item_id=3, text="Третий объект в последовательности"),
        DummyItem(item_id=4, text="Четвертый финальный объект"),
    ]

    counter = TokenCounter()
    one_item_tokens = counter.count(items[0].text)
    limit = one_item_tokens + 2

    chunker = TextChunker(ChunkingConfig(max_tokens=limit))
    batches = chunker.split_items(items, item_to_text=lambda x: x.text)

    assert len(batches) >= 2
    # Все исходные элементы присутствуют в батчах в исходном порядке
    unpacked = [item for b in batches for item in b]
    assert unpacked == items


def test_custom_token_counter_callable():
    """Проверка работы с пользовательской функцией подсчета токенов."""

    # Простейший счетчик по количеству слов
    def word_counter(s: str) -> int:
        return len(s.split())

    chunker = TextChunker(
        config=ChunkingConfig(max_tokens=5),
        token_counter=word_counter,
    )

    text = "раз два три четыре пять шесть семь восемь девять десять"
    chunks = chunker.split_text(text)

    assert len(chunks) == 2
    assert chunks[0] == "раз два три четыре пять"
    assert chunks[1] == "шесть семь восемь девять десять"
