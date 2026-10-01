"""Исключения для инструментов и подпроцессов chllm."""

from .base import CHLLMError


class CliToolError(CHLLMError):
    """Базовое исключение для ошибок выполнения внешних CLI-инструментов."""


class CliProcessError(CliToolError):
    """Исключение при завершении процесса с ненулевым кодом возврата."""

    def __init__(self, message: str, returncode: int, stderr: str = "", stdout: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


class CliTimeoutError(CliToolError):
    """Исключение при превышении таймаута выполнения процесса."""


class CliNotFoundError(CliToolError):
    """Исключение, если указанный исполняемый файл или команда не найдены."""
