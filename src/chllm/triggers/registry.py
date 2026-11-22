"""Реестр текстовых триггеров команд (TriggerRegistry)."""

import re
from typing import TYPE_CHECKING

from .domain import TextCommandTrigger

if TYPE_CHECKING:
    from ..parser import ToolCall


class TriggerRegistry:
    """Центральный реестр текстовых fallback-триггеров."""

    def __init__(self, triggers: list[TextCommandTrigger] | None = None) -> None:
        """Инициализирует реестр триггеров.

        Args:
            triggers: Начальный список триггеров.
        """
        self._triggers: list[TextCommandTrigger] = list(triggers or [])

    def register(
        self,
        name: str,
        pattern: str | re.Pattern[str],
        description: str = "",
        example: str = "",
    ) -> None:
        """Регистрирует новый текстовый триггер команды.

        Args:
            name: Имя целевого инструмента.
            pattern: Строковое регулярное выражение или скомпилированный паттерн.
            description: Описание команды для системного промпта.
            example: Пример синтаксиса для системного промпта.
        """
        compiled = re.compile(pattern, re.IGNORECASE) if isinstance(pattern, str) else pattern

        self._triggers.append(
            TextCommandTrigger(
                name=name,
                pattern=compiled,
                description=description,
                example=example,
            )
        )

    def parse_triggers(self, text: str) -> list["ToolCall"]:
        """Сканирует текст и преобразует все совпадения маркеров в объекты ToolCall.

        Args:
            text: Анализируемый текст ответа модели.

        Returns:
            Список извлеченных ToolCall.
        """
        from ..parser import ToolCall

        calls: list[ToolCall] = []
        for trigger in self._triggers:
            for match in trigger.pattern.finditer(text):
                args = match.groupdict()
                clean_args = {k: v.strip() if isinstance(v, str) else v for k, v in args.items()}
                calls.append(ToolCall(name=trigger.name, args=clean_args))
        return calls

    def strip_triggers(self, text: str) -> str:
        """Удаляет маркеры зарегистрированных триггеров из текста.

        Args:
            text: Исходный текст.

        Returns:
            Очищенный текст без маркеров команд.
        """
        cleaned = text
        for trigger in self._triggers:
            cleaned = trigger.pattern.sub("", cleaned)

        # Удаляем лишние повторяющиеся пустые строки и пробелы
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def format_instructions(self) -> str:
        """Формирует компактный блок подсказок для включения в промпт.

        Returns:
            Строка с описанием поддерживаемых текстовых команд или пустая строка.
        """
        if not self._triggers:
            return ""

        lines = [
            "--- FALLBACK TEXT COMMANDS ---",
            "If you cannot call tools via function calling, you may output these text commands:",
        ]
        for trigger in self._triggers:
            syntax = trigger.example or f"[{trigger.name.upper()}]"
            desc = trigger.description or f"Trigger tool {trigger.name}"
            lines.append(f"- {syntax}: {desc}")

        return "\n".join(lines)

    @classmethod
    def create_default(cls) -> "TriggerRegistry":
        """Создает реестр со стандартным набором встроенных триггеров.

        Returns:
            Экземпляр TriggerRegistry со стандартными триггерами:
            - [LOAD_SKILL: <name>]
            - [PAUSE: <reason>]
            - [DELEGATE: <role> | <task>]
        """
        registry = cls()
        registry.register(
            name="load_skill",
            pattern=r"\[LOAD_SKILL:\s*(?P<name>[a-zA-Z0-9_-]+)\]",
            description="Load a dynamic skill on demand",
            example="[LOAD_SKILL: <name>]",
        )
        registry.register(
            name="pause",
            pattern=r"\[PAUSE:\s*(?P<reason>.*?)\]",
            description="Pause the execution loop with explanation",
            example="[PAUSE: <reason>]",
        )
        registry.register(
            name="delegate_task",
            pattern=r"\[DELEGATE:\s*(?P<role>[a-zA-Z0-9_-]+)\s*\|\s*(?P<task>.*?)\]",
            description="Delegate subtask to a specialized subagent",
            example="[DELEGATE: <role> | <task>]",
        )
        return registry
