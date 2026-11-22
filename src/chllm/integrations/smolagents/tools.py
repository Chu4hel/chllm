"""Базовые адаптеры и утилиты для инструментов smolagents."""

import asyncio
import inspect
import threading
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from smolagents import Tool
else:
    Tool = Any

try:
    from chutils.logger import setup_logger
except ImportError:
    import logging

    def setup_logger(name: str = "chllm") -> Any:
        """Создает резервный логгер, если chutils не установлен."""
        return logging.getLogger(name)


logger = setup_logger("chllm.integrations.smolagents.tools")

T = TypeVar("T")


class BaseSmolToolAdapter:
    """Базовый адаптер для построения инструментов smolagents с потокобезопасным вызовом."""

    @staticmethod
    def dispatch_coroutine(
        coro: Coroutine[Any, Any, T],
        loop: asyncio.AbstractEventLoop,
        timeout: float = 60.0,
    ) -> T:
        """Выполняет асинхронную корутину в целевом цикле событий и ожидает результат.

        Args:
            coro: Корутина для выполнения.
            loop: Активный цикл событий asyncio.
            timeout: Максимальное время ожидания выполнения.

        Returns:
            Результат выполнения корутины.

        Raises:
            TimeoutError: При превышении времени ожидания.
        """
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result(timeout=timeout)


def _submit_coroutine(
    coro: "Coroutine[Any, Any, Any]",
    loop: asyncio.AbstractEventLoop | None,
) -> Any:
    """Выполняет корутину и возвращает результат, безопасно для Python 3.11–3.15."""
    target_loop = loop
    if target_loop is None or target_loop.is_closed():
        try:
            target_loop = asyncio.get_running_loop()
        except RuntimeError:
            target_loop = None

    running_in_current = False
    try:
        if asyncio.get_running_loop() is target_loop:
            running_in_current = True
    except RuntimeError:
        running_in_current = False

    if running_in_current:
        result: dict[str, Any] = {}

        def _worker() -> None:
            new_loop = asyncio.new_event_loop()
            try:
                result["value"] = new_loop.run_until_complete(coro)
            except BaseException as exc:
                result["exc"] = exc
            finally:
                new_loop.close()

        worker_thread = threading.Thread(target=_worker, daemon=True)
        worker_thread.start()
        worker_thread.join()
        if "exc" in result:
            raise result["exc"]
        return result["value"]

    if target_loop is not None:
        fut = asyncio.run_coroutine_threadsafe(coro, target_loop)
        return fut.result()

    return asyncio.run(coro)


def agent_tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
) -> Any:
    """Декоратор динамической регистрации инструмента smolagents (Dynamic Tool Registry)."""
    try:
        import smolagents  # noqa: F401
    except ImportError as exc:
        raise ImportError("Компонент 'agent_tool' требует установленной библиотеки smolagents.") from exc

    def _build(f: Callable[..., Any]) -> Any:
        from smolagents import Tool
        from smolagents._function_type_hints_utils import get_json_schema
        from smolagents.tools import TypeHintParsingException

        schema = get_json_schema(f)["function"]

        if "return" not in schema:
            if len(schema["parameters"]["properties"]) == 0:
                schema["return"] = {"type": "null"}
            else:
                raise TypeHintParsingException(
                    "Tool return type not found: make sure your function has a return type hint!"
                )

        tool_name = name or schema["name"]
        tool_description = description or schema["description"]
        tool_inputs = schema["parameters"]["properties"]
        tool_output_type = schema["return"]["type"]

        is_async = asyncio.iscoroutinefunction(f)

        class _AgentTool(Tool):  # type: ignore[misc]
            def __init__(self) -> None:
                self.is_initialized = True
                self._func = f
                self._loop = loop

        _AgentTool.name = tool_name
        _AgentTool.description = tool_description
        _AgentTool.inputs = tool_inputs
        _AgentTool.output_type = tool_output_type

        sig = inspect.signature(f)
        new_sig = sig.replace(
            parameters=[
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                *sig.parameters.values(),
            ]
        )

        def _forward(self: Any, *args: Any, **kwargs: Any) -> Any:
            if is_async:
                coro = self._func(*args, **kwargs)
                return _submit_coroutine(coro, self._loop)
            return self._func(*args, **kwargs)

        _AgentTool.forward = _forward
        _AgentTool.forward.__signature__ = new_sig
        _AgentTool.forward.__annotations__ = {
            "self": Any,
            **getattr(f, "__annotations__", {}),
        }

        return _AgentTool()

    if func is not None:
        return _build(func)

    def decorator(f: Callable[..., Any]) -> Any:
        return _build(f)

    return decorator
