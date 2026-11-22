"""Оболочка для вызова внешних CLI-инструментов (CliTool)."""

import asyncio
import json
import re
import shlex
import shutil
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..exceptions import CliNotFoundError, CliProcessError, CliTimeoutError
from ..parser import RobustLLMParser

ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class CliInputMode(StrEnum):
    """Режим передачи входных аргументов во внешний процесс."""

    STDIN_JSON = "stdin_json"
    CLI_ARGS = "cli_args"


class CliToolConfig(BaseModel):
    """Конфигурация вызова внешнего CLI инструмента.

    Attributes:
        command: Базовая команда или путь к бинарнику (список аргументов или строка).
        cwd: Рабочая директория процесса.
        env: Дополнительные переменные окружения.
        timeout: Таймаут выполнения в секундах.
        input_mode: Способ передачи аргументов (STDIN_JSON или CLI_ARGS).
        max_output_length: Максимальная длина вывода для защиты контекста LLM.
        strip_ansi: Удалять ли ANSI escape-коды форматирования из вывода.
        parse_json_response: Пытаться ли извлечь валидный JSON из вывода через RobustLLMParser.
        raise_on_error: Бросать ли исключение при ошибке процесса вместо возврата строки с ошибкой.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    command: list[str] | str = Field(..., description="Команда или список аргументов")
    cwd: str | Path | None = Field(default=None, description="Рабочая директория")
    env: dict[str, str] | None = Field(default=None, description="Переменные окружения")
    timeout: float = Field(default=30.0, description="Таймаут выполнения в секундах")
    input_mode: CliInputMode = Field(
        default=CliInputMode.STDIN_JSON,
        description="Режим передачи аргументов (stdin_json или cli_args)",
    )
    max_output_length: int = Field(default=4000, description="Максимальная длина вывода в символах")
    strip_ansi: bool = Field(default=True, description="Очищать ANSI escape-коды")
    parse_json_response: bool = Field(
        default=False,
        description="Извлекать структурированный JSON из вывода",
    )
    raise_on_error: bool = Field(
        default=False,
        description="Бросать исключение при сбоях вместо возврата строки с ошибкой",
    )


class CliTool:
    """Обертка над внешним CLI-процессом, совместимая с AgentOrchestrator."""

    def __init__(
        self,
        name: str,
        config: CliToolConfig | dict[str, Any],
        description: str = "",
    ) -> None:
        """Инициализирует CLI-инструмент.

        Args:
            name: Системное имя инструмента (для вызова агентом).
            config: Экземпляр CliToolConfig или словарь с параметрами конфигурации.
            description: Человекочитаемое описание назначения инструмента.
        """
        self.name = name
        self.description = description
        if isinstance(config, dict):
            self.config = CliToolConfig(**config)
        else:
            self.config = config
        self._parser = RobustLLMParser()

    async def __call__(self, **kwargs: Any) -> str:
        """Точка входа инструмента (совместима с сигнатурой Callable в orchestrator).

        Args:
            **kwargs: Аргументы, переданные языковой моделью.

        Returns:
            Результат выполнения в виде строки.
        """
        return await self.execute(**kwargs)

    async def execute(self, **kwargs: Any) -> str:
        """Выполняет внешний процесс с безопасной передачей аргументов и обработкой вывода.

        Args:
            **kwargs: Аргументы для процесса.

        Returns:
            Строковый результат работы команды или сообщение об ошибке.

        Raises:
            CliNotFoundError: Если исполняемый файл не найден и raise_on_error=True.
            CliTimeoutError: При превышении таймаута и raise_on_error=True.
            CliProcessError: При ненулевом коде возврата и raise_on_error=True.
        """
        cmd_args = self._resolve_command(kwargs)
        stdin_bytes = self._prepare_stdin(kwargs)

        binary = cmd_args[0]
        # Проверяем наличие бинарника через shutil.which, если это не прямой существующий файл
        resolved_bin = shutil.which(binary)
        if resolved_bin is not None:
            cmd_args[0] = resolved_bin
        elif not Path(binary).exists():
            msg = f"Command '{binary}' not found in PATH or filesystem."
            if self.config.raise_on_error:
                raise CliNotFoundError(msg)
            return f"Tool Execution Failed: {msg}"

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.config.cwd) if self.config.cwd else None,
                env=self.config.env,
            )

            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(input=stdin_bytes),
                timeout=self.config.timeout,
            )
        except TimeoutError as e:
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass
            msg = f"Process timed out after {self.config.timeout}s"
            if self.config.raise_on_error:
                raise CliTimeoutError(msg) from e
            return f"Tool Execution Failed: {msg}"
        except FileNotFoundError as e:
            msg = f"Command '{binary}' could not be executed: {e}"
            if self.config.raise_on_error:
                raise CliNotFoundError(msg) from e
            return f"Tool Execution Failed: {msg}"
        except Exception as e:
            msg = f"Unexpected error during process execution: {type(e).__name__}: {e}"
            if self.config.raise_on_error:
                raise
            return f"Tool Execution Failed: {msg}"

        stdout_text = stdout_data.decode("utf-8", errors="replace")
        stderr_text = stderr_data.decode("utf-8", errors="replace")

        if self.config.strip_ansi:
            stdout_text = ANSI_ESCAPE_PATTERN.sub("", stdout_text)
            stderr_text = ANSI_ESCAPE_PATTERN.sub("", stderr_text)

        if process.returncode != 0:
            err_details = stderr_text.strip() or stdout_text.strip() or f"exit code {process.returncode}"
            truncated_err = self._truncate(err_details)
            msg = f"Process failed with exit code {process.returncode}: {truncated_err}"
            if self.config.raise_on_error:
                raise CliProcessError(
                    msg,
                    returncode=process.returncode or -1,
                    stderr=stderr_text,
                    stdout=stdout_text,
                )
            return f"Tool Execution Failed (exit code {process.returncode}): {truncated_err}"

        # Успешное выполнение: обработка вывода
        output = stdout_text.strip()

        if self.config.parse_json_response:
            try:
                parsed_json = self._parser.parse_raw(output)
                output = json.dumps(parsed_json, indent=2, ensure_ascii=False)
            except Exception:
                # Если JSON извлечь не удалось, сохраняем исходный текст
                pass

        return self._truncate(output)

    def _resolve_command(self, kwargs: dict[str, Any]) -> list[str]:
        """Формирует список аргументов команды в зависимости от input_mode."""
        base_cmd: list[str]
        if isinstance(self.config.command, str):
            base_cmd = shlex.split(self.config.command, posix=False)
        else:
            base_cmd = list(self.config.command)

        if self.config.input_mode == CliInputMode.CLI_ARGS:
            args = list(base_cmd)
            for key, val in kwargs.items():
                flag = f"--{key.replace('_', '-')}"
                if isinstance(val, bool):
                    if val:
                        args.append(flag)
                elif isinstance(val, (list, tuple, Sequence)) and not isinstance(val, (str, bytes)):
                    for item in val:
                        args.extend([flag, str(item)])
                else:
                    args.extend([flag, str(val)])
            return args

        return list(base_cmd)

    def _prepare_stdin(self, kwargs: dict[str, Any]) -> bytes | None:
        """Формирует данные для передачи в stdin процесса."""
        if self.config.input_mode == CliInputMode.STDIN_JSON:
            return json.dumps(kwargs, ensure_ascii=False).encode("utf-8")
        return None

    def _truncate(self, text: str) -> str:
        """Ограничивает длину строки в соответствии с max_output_length."""
        if len(text) <= self.config.max_output_length:
            return text

        cutoff = self.config.max_output_length
        return f"{text[:cutoff]}\n[... Output truncated: {len(text)} characters total ...]"
