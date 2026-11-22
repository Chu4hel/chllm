"""Тесты интерактивной очереди Human-in-the-Loop в chllm.agentic."""

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel, Field

from chllm.agentic import (
    AgentTask,
    AgentTaskStatus,
    AgentTaskType,
    InteractiveTaskQueue,
)


class DummyScoreResult(BaseModel):
    """Тестовая схема результата."""

    score: int = Field(..., ge=1, le=10)
    comment: str


@pytest.mark.asyncio
async def test_interactive_queue_enqueue_and_resolve() -> None:
    """Проверяет постановку задачи и ее текстовое разрешение."""
    queue = InteractiveTaskQueue()

    async def _resolve_later() -> None:
        await asyncio.sleep(0.05)
        queue.resolve_task("task_1", "Ответ оператора")

    resolve_task = asyncio.create_task(_resolve_later())

    result = await queue.enqueue_task(
        task_id="task_1",
        title="Тестовая задача",
        prompt="Сделай что-нибудь",
        task_type=AgentTaskType.GENERAL,
    )
    await resolve_task

    assert result == "Ответ оператора"
    task = queue.get_task("task_1")
    assert task is not None
    assert task.status == AgentTaskStatus.RESOLVED


@pytest.mark.asyncio
async def test_interactive_queue_schema_validation() -> None:
    """Проверяет автоматическую валидацию по зарегистрированной Pydantic-схеме."""
    queue = InteractiveTaskQueue()
    queue.register_schema("DummyScore", DummyScoreResult)

    async def _resolve_json() -> None:
        await asyncio.sleep(0.05)
        json_reply = '```json\n{"score": 9, "comment": "Отличный результат"}\n```'
        queue.resolve_task("task_score", json_reply)

    resolve_task = asyncio.create_task(_resolve_json())

    result = await queue.enqueue_task(
        task_id="task_score",
        title="Оценка резюме",
        prompt="Оцени качество",
        schema_name="DummyScore",
        task_type=AgentTaskType.SCORE,
    )
    await resolve_task

    assert isinstance(result, DummyScoreResult)
    assert result.score == 9
    assert result.comment == "Отличный результат"


@pytest.mark.asyncio
async def test_interactive_queue_cancel_task() -> None:
    """Проверяет отмену ожидающей задачи."""
    queue = InteractiveTaskQueue()

    async def _cancel_later() -> None:
        await asyncio.sleep(0.05)
        queue.cancel_task("task_cancel")

    cancel_bg_task = asyncio.create_task(_cancel_later())

    with pytest.raises(asyncio.CancelledError):
        await queue.enqueue_task(
            task_id="task_cancel",
            title="Отменяемая задача",
            prompt="Промпт",
        )
    await cancel_bg_task

    task = queue.get_task("task_cancel")
    assert task is not None
    assert task.status == AgentTaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_interactive_queue_callbacks() -> None:
    """Проверяет вызовы коллбэков на добавление и завершение задачи."""
    queue = InteractiveTaskQueue()

    added_tasks: list[AgentTask] = []
    resolved_tasks: list[tuple[str, Any]] = []

    queue.subscribe(
        on_added=lambda t: added_tasks.append(t),
        on_resolved=lambda tid, res: resolved_tasks.append((tid, res)),
    )

    async def _resolve() -> None:
        await asyncio.sleep(0.05)
        queue.resolve_task("task_cb", "ok")

    resolve_bg_task = asyncio.create_task(_resolve())

    await queue.enqueue_task("task_cb", "Title", "Prompt")
    await resolve_bg_task

    assert len(added_tasks) == 1
    assert added_tasks[0].id == "task_cb"
    assert len(resolved_tasks) == 1
    assert resolved_tasks[0] == ("task_cb", "ok")
