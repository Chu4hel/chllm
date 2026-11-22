"""Тесты сжатия контекста диалога и директив лимита шагов."""

from chllm.context import build_step_limit_instruction, compress_history_messages


def test_compress_history_messages_short() -> None:
    """Проверяет, что короткая история сообщений не сжимается."""
    short = ["system prompt", "user query", "assistant reply"]
    assert compress_history_messages(short, keep_recent=4) == short


def test_compress_history_messages_long() -> None:
    """Проверяет сжатие старых длинных сообщений при сохранении крайних."""
    long_history = [
        "step 0 (system)",
        "a" * 500,
        "b" * 500,
        "recent 1",
        "recent 2",
    ]
    compressed = compress_history_messages(long_history, keep_recent=2, max_old_chars=100, preview_chars=50)
    assert len(compressed) == 5
    assert "[...сжато для экономии контекста" in compressed[1]
    assert "[...сжато для экономии контекста" in compressed[2]
    assert compressed[0] == "step 0 (system)"
    assert compressed[-1] == "recent 2"


def test_build_step_limit_instruction() -> None:
    """Проверяет генерацию текстовых инструкций лимита шагов."""
    msg_unlimited = build_step_limit_instruction(0)
    assert "Ограничение на количество шагов снято" in msg_unlimited

    msg_none = build_step_limit_instruction(None)
    assert "Ограничение на количество шагов снято" in msg_none

    msg_limited = build_step_limit_instruction(10)
    assert "максимум 10 шагов" in msg_limited
