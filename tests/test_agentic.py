"""Тесты для мультиагентной подсистемы (chllm.agentic)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from chllm.agentic import (
    AgentMessage,
    AgentSession,
    ConversationPool,
    ForkMode,
    InMemorySessionStorage,
    SessionMessageBus,
    SessionSnapshot,
)
from chllm.exceptions import SessionNotFoundError
from chllm.orchestrator import AgentOrchestrator


@pytest.fixture
def mock_provider():
    """Фикстура мок-провайдера LLM."""
    provider = MagicMock()
    provider.execute = AsyncMock()
    provider.pack_single_prompt = lambda p: p
    return provider


# --- 1. Тесты хранилища сессий (InMemorySessionStorage) ---


@pytest.mark.asyncio
async def test_in_memory_session_storage():
    """Проверка CRUD операций в InMemorySessionStorage."""
    storage = InMemorySessionStorage()

    s1 = SessionSnapshot(session_id="s1", role="Translator", system_prompt="Rule 1")
    s2 = SessionSnapshot(session_id="s2", parent_id="s1", role="Sub", system_prompt="Rule 2")

    await storage.save_session(s1)
    await storage.save_session(s2)

    # get_session
    loaded_s1 = await storage.get_session("s1")
    assert loaded_s1 is not None
    assert loaded_s1.role == "Translator"

    # list_sessions
    all_sessions = await storage.list_sessions()
    assert len(all_sessions) == 2

    child_sessions = await storage.list_sessions(parent_id="s1")
    assert len(child_sessions) == 1
    assert child_sessions[0].session_id == "s2"

    # delete_session
    deleted = await storage.delete_session("s1")
    assert deleted is True
    assert await storage.get_session("s1") is None
    assert await storage.delete_session("non_existent") is False


# --- 2. Тесты шины сообщений (SessionMessageBus) ---


@pytest.mark.asyncio
async def test_session_message_bus():
    """Проверка доставки, ожидания и чтения сообщений через шину."""
    bus = SessionMessageBus()

    # 1. Отправка и получение
    msg1 = AgentMessage(sender_id="agent_a", recipient_id="agent_b", content="Hello B")
    await bus.send_message(msg1)

    assert bus.has_messages("agent_b") is True
    assert bus.has_messages("agent_a") is False

    received = await bus.receive_message("agent_b", timeout=1.0)
    assert received is not None
    assert received.sender_id == "agent_a"
    assert received.content == "Hello B"

    # 2. Таймаут при пустом ящике
    timeout_res = await bus.receive_message("agent_b", timeout=0.1)
    assert timeout_res is None

    # 3. Накопление и read_inbox
    await bus.send_message(AgentMessage(sender_id="agent_a", recipient_id="agent_c", content="Msg 1"))
    await bus.send_message(AgentMessage(sender_id="agent_b", recipient_id="agent_c", content="Msg 2"))

    inbox = bus.read_inbox("agent_c")
    assert len(inbox) == 2
    assert inbox[0].content == "Msg 1"
    assert inbox[1].content == "Msg 2"
    assert bus.has_messages("agent_c") is False


# --- 3. Тесты AgentSession и ForkMode ---


@pytest.mark.asyncio
async def test_agent_session_execution_and_snapshot(mock_provider):
    """Проверка выполнения запроса в сессии и фиксации истории."""
    mock_provider.execute.return_value = "Translation completed."
    orchestrator = AgentOrchestrator(mock_provider)

    session = AgentSession(
        session_id="sess_1",
        orchestrator=orchestrator,
        role="Translator",
        system_prompt="Be polite.",
    )

    result = await session.execute("Hello world")
    assert result == "Translation completed."
    assert len(session.history) == 2
    assert session.history[0] == {"role": "user", "content": "Hello world"}
    assert session.history[1] == {"role": "assistant", "content": "Translation completed."}

    # Snapshot
    snap = session.to_snapshot()
    assert snap.session_id == "sess_1"
    assert snap.role == "Translator"
    assert len(snap.history) == 2

    # Восстановление
    restored = AgentSession.from_snapshot(snap, orchestrator=orchestrator)
    assert restored.session_id == "sess_1"
    assert len(restored.history) == 2


def test_conversation_pool_fork_modes(mock_provider):
    """Проверка режимов форка сессий: ISOLATED, INHERIT_PROMPT, FORK_HISTORY."""
    orchestrator = AgentOrchestrator(mock_provider)
    pool = ConversationPool(orchestrator=orchestrator)

    parent = pool.create_session(
        session_id="parent_1",
        role="Lead",
        system_prompt="Base rules.",
    )
    parent.history = [{"role": "user", "content": "Parent question"}, {"role": "assistant", "content": "Parent answer"}]

    # 1. ISOLATED: чистый лист
    iso_child = pool.fork_session(
        parent_session_id="parent_1",
        mode=ForkMode.ISOLATED,
        new_role="Worker",
        system_prompt="Custom worker rule.",
    )
    assert iso_child.role == "Worker"
    assert iso_child.parent_id == "parent_1"
    assert iso_child.system_prompt == "Custom worker rule."
    assert len(iso_child.history) == 0

    # 2. INHERIT_PROMPT: системный промпт наследуется, история пустая
    prompt_child = pool.fork_session(
        parent_session_id="parent_1",
        mode=ForkMode.INHERIT_PROMPT,
        new_role="Specialist",
        system_prompt="Additional specialty.",
    )
    assert "Base rules." in prompt_child.system_prompt
    assert "Additional specialty." in prompt_child.system_prompt
    assert len(prompt_child.history) == 0

    # 3. FORK_HISTORY: системный промпт и снимок истории копируются
    history_child = pool.fork_session(
        parent_session_id="parent_1",
        mode=ForkMode.FORK_HISTORY,
    )
    assert "Base rules." in history_child.system_prompt
    assert len(history_child.history) == 2
    assert history_child.history[0]["content"] == "Parent question"

    # Несуществующий родитель
    with pytest.raises(SessionNotFoundError):
        pool.fork_session(parent_session_id="unknown_parent")


# --- 4. Тесты RPC-делегирования и встроенных инструментов модели ---


@pytest.mark.asyncio
async def test_conversation_pool_rpc_delegate(mock_provider):
    """Проверка синхронного RPC-делегирования подзадачи субагенту."""
    # Субагент решает задачу
    mock_provider.execute.return_value = "Rhyme: cat / hat"
    orchestrator = AgentOrchestrator(mock_provider)
    pool = ConversationPool(orchestrator=orchestrator)

    parent = pool.create_session(role="MainTranslator")

    result = await pool.delegate(
        parent_id=parent.session_id,
        role="Poet",
        task="Find rhyme for cat",
        mode=ForkMode.ISOLATED,
    )

    assert result == "Rhyme: cat / hat"

    # Проверяем, что снимок дочерней сессии сохранился в storage
    saved_subsessions = await pool.storage.list_sessions(parent_id=parent.session_id)
    assert len(saved_subsessions) == 1
    assert saved_subsessions[0].role == "Poet"


@pytest.mark.asyncio
async def test_llm_calls_delegate_task_tool(mock_provider):
    """Тест: LLM самостоятельно вызывает инструмент delegate_task в диалоговом цикле."""
    # 1. Главный агент вызывает инструмент `delegate_task`
    # 2. Субагент (вызов 2) выполняет подзадачу
    # 3. Главный агент (вызов 3) формирует финальный ответ
    mock_provider.execute.side_effect = [
        '{"name": "delegate_task", "args": {"role": "FactChecker", "task": "Is RenPy Python-based?"}}',
        "Yes, RenPy is built on top of Python.",
        "RenPy написан на Python.",
    ]

    orchestrator = AgentOrchestrator(mock_provider)
    pool = ConversationPool(orchestrator=orchestrator)

    main_session = pool.create_session(role="Assistant")
    final_answer = await main_session.execute("Расскажи о движке RenPy")

    assert final_answer == "RenPy написан на Python."
    assert mock_provider.execute.call_count == 3

    # Проверяем, что результат субагента попал в tool results главного агента
    third_prompt = mock_provider.execute.call_args_list[2][0][0]
    assert "Result of delegate_task: Yes, RenPy is built on top of Python." in third_prompt


@pytest.mark.asyncio
async def test_llm_message_bus_tools(mock_provider):
    """Тест: LLM отправляет сообщение через send_message и читает почту через read_inbox."""
    orchestrator = AgentOrchestrator(mock_provider)
    pool = ConversationPool(orchestrator=orchestrator)

    session_a = pool.create_session(session_id="agent_1", role="Coordinator")
    session_b = pool.create_session(session_id="agent_2", role="Worker")

    # 1. Агент 1 вызывает инструмент send_message
    mock_provider.execute.side_effect = [
        f'{{"name": "send_message", "args": {{"recipient_id": "{session_b.session_id}", "content": "Start task"}}}}',
        "Сообщение отправлено.",
    ]

    await session_a.execute("Отправь задачу агенту 2")
    assert pool.bus.has_messages(session_b.session_id) is True

    # 2. Агент 2 вызывает инструмент read_inbox
    mock_provider.execute.side_effect = [
        '{"name": "read_inbox", "args": {}}',
        "Задачу получил, приступаю.",
    ]

    await session_b.execute("Проверь новые сообщения")
    second_call_prompt = mock_provider.execute.call_args_list[3][0][0]
    assert "Inbox (1 messages):" in second_call_prompt
    assert "From agent_1: Start task" in second_call_prompt
