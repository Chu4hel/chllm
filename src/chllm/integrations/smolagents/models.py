"""Адаптеры моделей smolagents для работы в режимах WebChat (HITL) и Buffer (Zero-API)."""

import asyncio
import concurrent.futures
import json
import re
import uuid
from collections.abc import Coroutine
from typing import Any

from pydantic import BaseModel
from smolagents import OpenAIServerModel, Tool
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
)

from ...agentic.domain import AgentMode, AgentTaskType, IInteractiveQueue
from ...buffer import FileTaskQueue
from ...context import build_step_limit_instruction, compress_history_messages

try:
    from chutils.logger import setup_logger
except ImportError:
    import logging

    def setup_logger(name: str = "chllm") -> Any:
        """Создает резервный логгер, если chutils не установлен."""
        return logging.getLogger(name)


logger = setup_logger("chllm.integrations.smolagents.models")


def _submit_coroutine(
    coro: Coroutine[Any, Any, Any],
    loop: asyncio.AbstractEventLoop | None,
) -> "concurrent.futures.Future[Any]":
    """Отправляет корутину в целевой event loop и возвращает Future с результатом.

    Начиная с Python 3.14 ``asyncio.get_event_loop()`` больше не создаёт цикл событий
    неявно и бросает RuntimeError, поэтому при отсутствии запущенного цикла корутина
    выполняется синхронно через ``asyncio.run`` в текущем потоке.

    Args:
        coro: Корутина для выполнения.
        loop: Целевой цикл событий (может быть None или уже закрытым).

    Returns:
        Future с результатом выполнения корутины.
    """
    target_loop = loop
    if target_loop is None or target_loop.is_closed():
        try:
            target_loop = asyncio.get_running_loop()
        except RuntimeError:
            target_loop = None

    if target_loop is not None:
        return asyncio.run_coroutine_threadsafe(coro, target_loop)

    future: concurrent.futures.Future[Any] = concurrent.futures.Future()
    try:
        future.set_result(asyncio.run(coro))
    except BaseException as exc:
        future.set_exception(exc)
    return future


class WebChatSmolModel(Model):
    """Адаптер модели smolagents для интерактивного режима через веб-чат (Human-in-the-Loop).

    Преобразует шаги рассуждения ToolCallingAgent в задачи интерактивной очереди,
    ожидает ручной ввод вызова инструмента или финального ответа от пользователя в UI
    и возвращает структурированный ChatMessage.
    """

    def __init__(
        self,
        interactive_queue: IInteractiveQueue,
        loop: asyncio.AbstractEventLoop | None = None,
        timeout_seconds: float = 600.0,
        task_type: AgentTaskType = AgentTaskType.SELF_HEALING,
        task_title_prefix: str = "Smolagents Interactive",
        max_steps_limit: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Инициализирует веб-чат адаптер модели.

        Args:
            interactive_queue: Сервис очереди интерактивных задач (HITL).
            loop: Ссылка на запущенный asyncio event loop.
            timeout_seconds: Максимальное время ожидания ответа пользователя в секундах.
            task_type: Тип задачи для отображения в очереди веб-чата.
            task_title_prefix: Префикс заголовка задачи в веб-чате.
            max_steps_limit: Опциональный лимит шагов агента.
            **kwargs: Дополнительные параметры базовой модели Model.
        """
        super().__init__(**kwargs)
        self.queue = interactive_queue
        self.loop = loop
        self.timeout_seconds = timeout_seconds
        self.task_type = task_type
        self.task_title_prefix = task_title_prefix
        self.max_steps_limit = max_steps_limit
        self.step_counter = 0

    def generate(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        stop_sequences: list[str] | None = None,
        grammar: dict[str, Any] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Генерирует следующий шаг агента через запрос к человеку в интерактивной очереди.

        Args:
            messages: История сообщений диалога агента.
            stop_sequences: Список стоп-последовательностей.
            grammar: Опциональная грамматика генерации.
            tools_to_call_from: Доступные агенту инструменты для вызова.
            **kwargs: Дополнительные аргументы.

        Returns:
            Сконструированный объект ChatMessage с вызовом инструмента или финальным ответом.
        """
        self.step_counter += 1

        tools_info: list[dict[str, Any]] = []
        if tools_to_call_from:
            for t in tools_to_call_from:
                tools_info.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputs": t.inputs,
                        "output_type": getattr(t, "output_type", "string"),
                    }
                )

        tools_json = json.dumps(tools_info, ensure_ascii=False, indent=2)

        history_repr: list[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                r = msg.get("role", "user")
                c = msg.get("content", "")
                if isinstance(c, list):
                    c = "\n".join(item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in c)
                history_repr.append(f"[{str(r).upper()}]:\n{str(c).strip()}")
            elif hasattr(msg, "role"):
                role_val = getattr(msg.role, "value", str(msg.role))
                raw_c = getattr(msg, "content", "")
                if isinstance(raw_c, list):
                    c = "\n".join(
                        item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in raw_c
                    )
                else:
                    c = str(raw_c) if raw_c is not None else ""

                tool_calls_repr = ""
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    calls_strs = []
                    for tc in msg.tool_calls:
                        fn = getattr(tc, "function", None)
                        if fn:
                            fn_name = getattr(fn, "name", "unknown")
                            fn_args = getattr(fn, "arguments", {})
                            calls_strs.append(f"ToolCall {fn_name}({fn_args})")
                    if calls_strs:
                        tool_calls_repr = "\n" + "\n".join(calls_strs)

                history_repr.append(f"[{role_val.upper()}]:\n{c.strip()}{tool_calls_repr}")

        compressed_history = compress_history_messages(history_repr)
        full_context_text = "\n\n---\n\n".join(compressed_history)

        step_indicator = f" (Шаг {self.step_counter}"
        if self.max_steps_limit and self.max_steps_limit > 0:
            step_indicator += f" из {self.max_steps_limit})"
        else:
            step_indicator += ")"

        task_id = f"smol_step_{uuid.uuid4().hex[:8]}"
        title = f"{self.task_title_prefix}{step_indicator}"

        step_limit_msg = build_step_limit_instruction(self.max_steps_limit)

        prompt_for_user = (
            f"### Агент smolagents выполняет задачу{step_indicator}\n\n"
            f"> [!TIP]\n"
            f"> {step_limit_msg}\n\n"
            f"**Доступные инструменты (JSON-схема):**\n```json\n{tools_json}\n```\n\n"
            f"**Текущая история рассуждений и шагов:**\n{full_context_text}\n\n"
            "**Инструкция по ответу:**\n"
            "- Чтобы вызвать инструмент, отправьте JSON вида:\n"
            '  `{"action": "call_tool", "tool_name": "<имя_инструмента>", "arguments": { ... }}`\n'
            "- Чтобы завершить работу и дать финальный ответ, отправьте JSON вида:\n"
            '  `{"action": "final_answer", "answer": "<текст ответа>"}`\n'
            "  или просто обычный текстовый ответ (он будет расценен как финальный)."
        )

        fut = self._enqueue_in_loop(
            task_id=task_id,
            title=title,
            prompt=prompt_for_user,
            task_type=self.task_type,
            context={"step": self.step_counter, "tools": tools_info},
        )

        try:
            raw_response = fut.result(timeout=self.timeout_seconds)
        except Exception as e:
            logger.error("Таймаут или ошибка ожидания ответа пользователя: %s", e)
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            error_call = ChatMessageToolCall(
                function=ChatMessageToolCallFunction(
                    name="final_answer",
                    arguments={"answer": f"Ошибка ожидания ответа оператора: {e}"},
                ),
                id=call_id,
                type="function",
            )
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Оператор не ответил вовремя.",
                tool_calls=[error_call],
            )

        return self._parse_user_response_to_chat_message(str(raw_response))

    def _enqueue_in_loop(
        self,
        task_id: str,
        title: str,
        prompt: str,
        task_type: AgentTaskType,
        context: dict[str, Any],
    ) -> concurrent.futures.Future[BaseModel | str]:
        """Помещает задачу в очередь интерактивного веб-чата через целевой event loop."""
        coro = self.queue.enqueue_task(
            task_id=task_id,
            title=title,
            prompt=prompt,
            schema_name=None,
            task_type=task_type,
            context=context,
        )

        return _submit_coroutine(coro, self.loop)

    def _parse_user_response_to_chat_message(self, raw: str) -> ChatMessage:
        """Парсит полученный ответ оператора в ChatMessage с ToolCall или final_answer."""
        text = raw.strip()
        json_obj = None

        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        candidate = match.group(1).strip() if match else text

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                json_obj = parsed
        except json.JSONDecodeError:
            pass

        if json_obj:
            action = json_obj.get("action")
            if action == "call_tool" or "tool_name" in json_obj:
                tool_name = json_obj.get("tool_name") or json_obj.get("name")
                arguments = json_obj.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw_args": arguments}

                call_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_call = ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(
                        name=str(tool_name),
                        arguments=arguments,
                    ),
                    id=call_id,
                    type="function",
                )
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=None,
                    tool_calls=[tool_call],
                )

            if action == "final_answer" or "answer" in json_obj:
                answer = json_obj.get("answer", str(json_obj))
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                final_call = ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(
                        name="final_answer",
                        arguments={"answer": answer},
                    ),
                    id=call_id,
                    type="function",
                )
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=str(answer),
                    tool_calls=[final_call],
                )

            if "name" in json_obj and "arguments" in json_obj:
                tool_name = json_obj["name"]
                arguments = json_obj["arguments"]
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_call = ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(
                        name=str(tool_name),
                        arguments=arguments,
                    ),
                    id=call_id,
                    type="function",
                )
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=None,
                    tool_calls=[tool_call],
                )

        call_id = f"call_{uuid.uuid4().hex[:8]}"
        final_call = ChatMessageToolCall(
            function=ChatMessageToolCallFunction(
                name="final_answer",
                arguments={"answer": text},
            ),
            id=call_id,
            type="function",
        )
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=text,
            tool_calls=[final_call],
        )


class BufferSmolModel(Model):
    """Адаптер модели smolagents для Zero-API взаимодействия через файловый буфер.

    Записывает каждый шаг рассуждения ToolCallingAgent в `.agent_inbox/<task_id>.json`
    и ожидает ответ локального LLM или IDE-агента в `.agent_outbox/<task_id>.json`.
    """

    def __init__(
        self,
        file_queue: FileTaskQueue,
        loop: asyncio.AbstractEventLoop | None = None,
        timeout_seconds: float = 600.0,
        task_title_prefix: str = "Zero-API Smolagent Step",
        max_steps_limit: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Инициализирует буферный адаптер модели smolagents.

        Args:
            file_queue: Файловый буфер очереди задач.
            loop: Ссылка на цикл событий.
            timeout_seconds: Максимальное время ожидания ответа.
            task_title_prefix: Префикс заголовка задачи.
            max_steps_limit: Опциональный лимит шагов.
            **kwargs: Дополнительные аргументы Model.
        """
        super().__init__(**kwargs)
        self.file_queue = file_queue
        self.loop = loop
        self.timeout_seconds = timeout_seconds
        self.task_title_prefix = task_title_prefix
        self.max_steps_limit = max_steps_limit
        self.step_counter = 0

    def generate(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        stop_sequences: list[str] | None = None,
        grammar: dict[str, Any] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Генерирует следующий шаг через запись задачи в буфер и ожидание в outbox.

        Args:
            messages: История сообщений диалога агента.
            stop_sequences: Список стоп-последовательностей.
            grammar: Опциональная грамматика генерации.
            tools_to_call_from: Доступные агенту инструменты для вызова.
            **kwargs: Дополнительные аргументы генерации.

        Returns:
            Сконструированный объект ChatMessage с вызовом инструмента или финальным ответом.
        """
        self.step_counter += 1

        tools_info: list[dict[str, Any]] = []
        if tools_to_call_from:
            for t in tools_to_call_from:
                tools_info.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputs": t.inputs,
                        "output_type": getattr(t, "output_type", "string"),
                    }
                )

        task_id = f"buffer_step_{uuid.uuid4().hex[:8]}"
        payload = {
            "step": self.step_counter,
            "max_steps_limit": self.max_steps_limit,
            "tools": tools_info,
            "messages": [
                m
                if isinstance(m, dict)
                else {
                    "role": str(getattr(m, "role", "")),
                    "content": str(getattr(m, "content", "")),
                }
                for m in messages
            ],
            "instruction": build_step_limit_instruction(self.max_steps_limit),
        }

        self.file_queue.push_task(
            task_id=task_id,
            task_type="SMOLAGENT_STEP",
            payload=payload,
        )

        try:
            coro = self.file_queue.wait_for_result(task_id, timeout=self.timeout_seconds)
            fut = _submit_coroutine(coro, self.loop)
            result_dict = fut.result(timeout=self.timeout_seconds)
        except Exception as e:
            logger.error("Таймаут или сбой ожидания Zero-API буфера для шага %s: %s", task_id, e)
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            error_call = ChatMessageToolCall(
                function=ChatMessageToolCallFunction(
                    name="final_answer",
                    arguments={"answer": f"Zero-API буфер не ответил: {e}"},
                ),
                id=call_id,
                type="function",
            )
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Zero-API буфер не ответил вовремя.",
                tool_calls=[error_call],
            )

        raw_output = (
            result_dict.get("result") or result_dict.get("answer") or json.dumps(result_dict, ensure_ascii=False)
        )
        return self._parse_buffer_response(raw_output)

    def _parse_buffer_response(self, raw: Any) -> ChatMessage:
        """Преобразует ответ из буфера в ChatMessage."""
        if isinstance(raw, dict):
            if "tool_name" in raw:
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_call = ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(
                        name=str(raw["tool_name"]),
                        arguments=raw.get("arguments", {}),
                    ),
                    id=call_id,
                    type="function",
                )
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=None,
                    tool_calls=[tool_call],
                )
            if "answer" in raw:
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                final_call = ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(
                        name="final_answer",
                        arguments={"answer": str(raw["answer"])},
                    ),
                    id=call_id,
                    type="function",
                )
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=str(raw["answer"]),
                    tool_calls=[final_call],
                )

        text = str(raw).strip()
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        final_call = ChatMessageToolCall(
            function=ChatMessageToolCallFunction(
                name="final_answer",
                arguments={"answer": text},
            ),
            id=call_id,
            type="function",
        )
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=text,
            tool_calls=[final_call],
        )


def create_agent_model(
    mode: AgentMode | str,
    api_model_id: str = "gpt-4o-mini",
    api_base: str | None = None,
    api_key: str | None = None,
    interactive_queue: IInteractiveQueue | None = None,
    file_queue: FileTaskQueue | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
    task_type: AgentTaskType = AgentTaskType.SELF_HEALING,
    task_title_prefix: str = "Smolagent Step",
    max_steps_limit: int | None = None,
) -> Model:
    """Создает адаптер модели smolagents для выбранного протокола (API, WebChat, Buffer).

    Args:
        mode: Протокол работы: 'api', 'web_chat' или 'buffer'.
        api_model_id: Идентификатор модели для режима прямого API.
        api_base: Базовый URL API (например, http://localhost:11434/v1 для Ollama).
        api_key: Ключ API.
        interactive_queue: Очередь интерактивного взаимодействия (для 'web_chat').
        file_queue: Файловый буфер задач (для 'buffer').
        loop: Цикл событий asyncio.
        task_type: Тип задачи для интерактивной очереди.
        task_title_prefix: Префикс заголовка задачи.
        max_steps_limit: Лимит шагов агента.

    Returns:
        Экземпляр модели Model smolagents.

    Raises:
        ValueError: При отсутствии необходимых очередей или неизвестном режиме.
    """
    mode_val = mode.value if isinstance(mode, AgentMode) else str(mode).lower()

    if mode_val in (AgentMode.API.value, "autonomous"):
        return OpenAIServerModel(
            model_id=api_model_id,
            api_base=api_base,
            api_key=api_key,
        )

    if mode_val in (AgentMode.WEB_CHAT.value, "manual_web"):
        if interactive_queue is None:
            raise ValueError("Для протокола WEB_CHAT требуется экземпляр IInteractiveQueue.")
        return WebChatSmolModel(
            interactive_queue=interactive_queue,
            loop=loop,
            task_type=task_type,
            task_title_prefix=task_title_prefix,
            max_steps_limit=max_steps_limit,
        )

    if mode_val in (AgentMode.BUFFER.value, "agent_buffer"):
        if file_queue is None:
            raise ValueError("Для протокола BUFFER (Zero-API) требуется экземпляр FileTaskQueue.")
        return BufferSmolModel(
            file_queue=file_queue,
            loop=loop,
            task_title_prefix=task_title_prefix,
            max_steps_limit=max_steps_limit,
        )

    raise ValueError(f"Неизвестный протокол агента: {mode}")
