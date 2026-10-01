from unittest.mock import AsyncMock, MagicMock

import pytest

from chllm.chunker import ChunkingConfig, TextChunker
from chllm.exceptions import CHLLMError
from chllm.orchestrator import Orchestrator
from chllm.summarizer import ChainedSummarizer, HierarchicalProcessor, MapReduceConfig


@pytest.fixture
def mock_orchestrator():
    """Фикстура мок-оркестратора."""
    provider = MagicMock()
    provider.execute = AsyncMock()
    provider.pack_single_prompt = lambda p: {"prompt": p}
    orchestrator = Orchestrator(provider)
    return orchestrator


@pytest.mark.asyncio
async def test_empty_input(mock_orchestrator):
    """Проверка возврата пустой строки при пустом вводе."""
    processor = HierarchicalProcessor(mock_orchestrator)
    assert await processor.process_text("") == ""
    assert await processor.process_text("   \n  ") == ""
    assert await processor.process_chunks([]) == ""


@pytest.mark.asyncio
async def test_single_chunk_optimization(mock_orchestrator, monkeypatch):
    """Проверка оптимизации для одного чанка без вызова Reduce."""
    config = MapReduceConfig(
        single_prompt_template="[SINGLE] {text}",
        map_prompt_template="[MAP] {text}",
        reduce_prompt_template="[REDUCE] {text}",
    )
    processor = HierarchicalProcessor(mock_orchestrator, config=config)

    monkeypatch.setattr(mock_orchestrator, "execute_single", AsyncMock(return_value="Single summary result"))

    progress_events = []

    def on_progress(phase, cur, total):
        progress_events.append((phase, cur, total))

    result = await processor.process_chunks(["Short text"], on_progress=on_progress)

    assert result == "Single summary result"
    mock_orchestrator.execute_single.assert_awaited_once_with("[SINGLE] Short text")
    assert progress_events == [("single", 1, 1)]


@pytest.mark.asyncio
async def test_map_reduce_parallel_flow(mock_orchestrator, monkeypatch):
    """Проверка параллельного Map-Reduce пайплайна."""
    config = MapReduceConfig(
        concurrency_limit=2,
        map_prompt_template="MAP: {text}",
        reduce_prompt_template="REDUCE:\n{text}",
        summary_prefix="Part {index}: {summary}",
    )
    processor = HierarchicalProcessor(mock_orchestrator, config=config)

    async def mock_execute(prompt: str) -> str:
        if prompt.startswith("MAP: Chunk 1"):
            return "Summary 1"
        if prompt.startswith("MAP: Chunk 2"):
            return "Summary 2"
        if "REDUCE:" in prompt:
            return "Final Combined Summary"
        return "Unknown"

    monkeypatch.setattr(mock_orchestrator, "execute_single", AsyncMock(side_effect=mock_execute))

    progress = []

    async def async_progress(phase, cur, total):
        progress.append((phase, cur, total))

    result = await processor.process_chunks(["Chunk 1", "Chunk 2"], on_progress=async_progress)

    assert result == "Final Combined Summary"
    assert ("reduce", 1, 1) in progress
    assert len([p for p in progress if p[0] == "map"]) == 2


@pytest.mark.asyncio
async def test_map_reduce_sequential_flow(mock_orchestrator, monkeypatch):
    """Проверка строго последовательного Map-Reduce при concurrency_limit=1."""
    config = MapReduceConfig(
        concurrency_limit=1,
        map_prompt_template="MAP: {{text}}",
        reduce_prompt_template="REDUCE: {{text}}",
    )
    processor = HierarchicalProcessor(mock_orchestrator, config=config)

    executed_prompts = []

    async def mock_execute(prompt: str) -> str:
        executed_prompts.append(prompt)
        if "Chunk 1" in prompt:
            return "Sum 1"
        if "Chunk 2" in prompt:
            return "Sum 2"
        return "Merged"

    monkeypatch.setattr(mock_orchestrator, "execute_single", AsyncMock(side_effect=mock_execute))

    result = await processor.process_chunks(["Chunk 1", "Chunk 2"])

    assert result == "Merged"
    assert len(executed_prompts) == 3
    assert executed_prompts[0] == "MAP: Chunk 1"
    assert executed_prompts[1] == "MAP: Chunk 2"
    assert "REDUCE:" in executed_prompts[2]


@pytest.mark.asyncio
async def test_hierarchical_multi_level_reduce(mock_orchestrator, monkeypatch):
    """Проверка рекурсивного сжатия Reduce, когда выжимки превышают chunk_size."""
    # Задаем очень маленький chunk_size, чтобы объединение выжимок вынудило рекурсивный Reduce
    config = MapReduceConfig(
        chunk_size=15,  # Очень маленький размер чанка в токенах
        concurrency_limit=2,
        map_prompt_template="MAP: {text}",
        reduce_prompt_template="REDUCE:\n{text}",
    )
    chunker = TextChunker(config=ChunkingConfig(max_tokens=15))
    processor = HierarchicalProcessor(mock_orchestrator, config=config, chunker=chunker)

    call_count = 0

    async def mock_execute(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if "MAP:" in prompt:
            # Длинная выжимка, которая займет заметное число токенов
            return "A long local summary that takes several tokens to describe events."
        return "Final hierarchical summary"

    monkeypatch.setattr(mock_orchestrator, "execute_single", AsyncMock(side_effect=mock_execute))

    chunks = ["Paragraph 1", "Paragraph 2", "Paragraph 3", "Paragraph 4"]
    result = await processor.process_chunks(chunks)

    assert result == "Final hierarchical summary"
    # Должно быть 4 вызова MAP + минимум 1 вызов многоуровневого REDUCE
    assert call_count >= 5


@pytest.mark.asyncio
async def test_fail_on_chunk_error(mock_orchestrator, monkeypatch):
    """Проверка выброса исключения при ошибке чанка и fail_on_chunk_error=True."""
    config = MapReduceConfig(fail_on_chunk_error=True)
    processor = HierarchicalProcessor(mock_orchestrator, config=config)

    async def failing_execute(prompt: str) -> str:
        if "Chunk 2" in prompt:
            raise RuntimeError("API quota exceeded")
        return "OK"

    monkeypatch.setattr(mock_orchestrator, "execute_single", AsyncMock(side_effect=failing_execute))

    with pytest.raises(CHLLMError) as exc_info:
        await processor.process_chunks(["Chunk 1", "Chunk 2"])

    assert "Ошибка обработки чанка" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ignore_chunk_error_when_flag_false(mock_orchestrator, monkeypatch):
    """Проверка продолжения работы при fail_on_chunk_error=False."""
    config = MapReduceConfig(fail_on_chunk_error=False, concurrency_limit=1)
    processor = HierarchicalProcessor(mock_orchestrator, config=config)

    async def failing_execute(prompt: str) -> str:
        if "Chunk 2" in prompt:
            raise RuntimeError("Temporary error")
        return "OK Summary"

    monkeypatch.setattr(mock_orchestrator, "execute_single", AsyncMock(side_effect=failing_execute))

    result = await processor.process_chunks(["Chunk 1", "Chunk 2"])
    # Так как осталась только одна успешная выжимка, она сразу возвращается
    assert result == "OK Summary"


@pytest.mark.asyncio
async def test_chained_summarizer_refine_mode(mock_orchestrator, monkeypatch):
    """Проверка работы ChainedSummarizer в последовательном режиме refine."""
    config = MapReduceConfig(
        map_prompt_template="FIRST: {text}",
        refine_prompt_template="REFINE: prev={previous_summary} new={text}",
    )
    summarizer = ChainedSummarizer(mock_orchestrator, config=config, chain_mode="refine")

    history = []

    async def mock_execute(prompt: str) -> str:
        history.append(prompt)
        if prompt.startswith("FIRST:"):
            return "Sum1"
        return f"Refined({prompt})"

    monkeypatch.setattr(mock_orchestrator, "execute_single", AsyncMock(side_effect=mock_execute))

    result = await summarizer.summarize_chunks(["Part A", "Part B", "Part C"])

    assert len(history) == 3
    assert history[0] == "FIRST: Part A"
    assert "prev=Sum1" in history[1]
    assert "new=Part B" in history[1]
    assert "new=Part C" in history[2]
    assert result.startswith("Refined(")


@pytest.mark.asyncio
async def test_summarizer_process_text(mock_orchestrator, monkeypatch):
    """Проверка автоматической нарезки текста через process_text / summarize."""
    config = MapReduceConfig(chunk_size=10, overlap_tokens=0)
    summarizer = ChainedSummarizer(mock_orchestrator, config=config)

    monkeypatch.setattr(mock_orchestrator, "execute_single", AsyncMock(return_value="Done"))

    long_text = "Sentence one. " * 30
    res = await summarizer.summarize(long_text)
    assert res == "Done"
    assert mock_orchestrator.execute_single.await_count >= 2
