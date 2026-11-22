"""Тесты файлового буфера Zero-API и провайдера BufferProvider в chllm.buffer."""

import asyncio
import json
import time
from pathlib import Path

import pytest

from chllm.buffer import BufferProvider, FileTaskQueue
from chllm.orchestrator import Orchestrator


def test_file_task_queue_push_and_get(tmp_path: Path) -> None:
    """Проверяет запись задачи в inbox и чтение результата из outbox."""
    inbox = tmp_path / "inbox"
    outbox = tmp_path / "outbox"
    queue = FileTaskQueue(inbox_dir=inbox, outbox_dir=outbox)

    task_id = "test_task_1"
    queue.push_task(
        task_id=task_id,
        task_type="TEST_OP",
        payload={"query": "hello world"},
    )

    # Проверяем файлы в inbox
    assert (inbox / f"{task_id}.json").exists()
    assert (inbox / f"{task_id}.md").exists()
    assert queue.has_pending_tasks()
    assert queue.list_pending_tasks() == [task_id]

    # До записи в outbox результат None
    assert queue.get_result(task_id) is None
    assert not queue.has_response(task_id)

    # Записываем результат в outbox
    (outbox / f"{task_id}.json").write_text(
        json.dumps({"result": "success", "data": 42}),
        encoding="utf-8",
    )

    assert queue.has_response(task_id)
    res = queue.get_result(task_id)
    assert res == {"result": "success", "data": 42}


@pytest.mark.asyncio
async def test_file_task_queue_wait_for_result(tmp_path: Path) -> None:
    """Проверяет асинхронное ожидание результата в outbox."""
    inbox = tmp_path / "inbox"
    outbox = tmp_path / "outbox"
    queue = FileTaskQueue(inbox_dir=inbox, outbox_dir=outbox)

    task_id = "async_task_1"
    queue.push_task(task_id, "OP", {"foo": "bar"})

    async def _write_response_later() -> None:
        await asyncio.sleep(0.1)
        (outbox / f"{task_id}.json").write_text(
            json.dumps({"result": "completed"}),
            encoding="utf-8",
        )

    task = asyncio.create_task(_write_response_later())

    res = await queue.wait_for_result(task_id, timeout=2.0, poll_interval=0.05)
    await task
    assert res == {"result": "completed"}


@pytest.mark.asyncio
async def test_file_task_queue_timeout(tmp_path: Path) -> None:
    """Проверяет выброс TimeoutError при отсутствии ответа."""
    queue = FileTaskQueue(inbox_dir=tmp_path / "inbox", outbox_dir=tmp_path / "outbox")

    with pytest.raises(TimeoutError):
        await queue.wait_for_result("missing_task", timeout=0.1, poll_interval=0.02)


def test_file_task_queue_cleanup_stale_outbox(tmp_path: Path) -> None:
    """Проверяет архивацию и очистку устаревших файлов."""
    outbox = tmp_path / "outbox"
    queue = FileTaskQueue(
        inbox_dir=tmp_path / "inbox",
        outbox_dir=outbox,
        archive_days=1,
        purge_days=3,
    )

    old_file = outbox / "old_task.json"
    old_file.write_text("{}", encoding="utf-8")
    # Устанавливаем время модификации 2 дня назад
    two_days_ago = time.time() - (2 * 86400)
    import os

    os.utime(old_file, (two_days_ago, two_days_ago))

    counts = queue.cleanup_stale_outbox()
    assert counts["archived"] == 1
    assert not old_file.exists()
    assert (outbox / "archive" / "old_task.json").exists()


@pytest.mark.asyncio
async def test_buffer_provider_integration(tmp_path: Path) -> None:
    """Проверяет интеграцию BufferProvider с Orchestrator."""
    inbox = tmp_path / "inbox"
    outbox = tmp_path / "outbox"
    queue = FileTaskQueue(inbox_dir=inbox, outbox_dir=outbox)
    provider = BufferProvider(file_queue=queue, timeout=2.0, poll_interval=0.05)
    orchestrator = Orchestrator(provider=provider)

    async def _simulate_agent_response() -> None:
        # Ждем появления задачи в inbox
        start = time.time()
        while not queue.has_pending_tasks() and time.time() - start < 2.0:
            await asyncio.sleep(0.05)

        pending = queue.list_pending_tasks()
        assert len(pending) == 1
        tid = pending[0]

        # Записываем ответ агента в outbox
        (outbox / f"{tid}.json").write_text(
            json.dumps({"result": "Ответ от Zero-API ассистента"}),
            encoding="utf-8",
        )

    task = asyncio.create_task(_simulate_agent_response())

    answer = await orchestrator.execute_single("Привет, ассистент!")
    await task
    assert answer == "Ответ от Zero-API ассистента"
