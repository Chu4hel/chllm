"""Тесты для подсистемы текстовых fallback-триггеров (chllm.triggers)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from chllm.builder import PromptBuilder
from chllm.orchestrator import AgentOrchestrator
from chllm.parser import RobustLLMParser
from chllm.triggers import TriggerRegistry


@pytest.fixture
def mock_provider():
    """Фикстура мок-провайдера LLM."""
    provider = MagicMock()
    provider.execute = AsyncMock()
    provider.pack_single_prompt = lambda p: p
    return provider


def test_trigger_registry_default():
    """Проверка стандартного набора встроенных триггеров."""
    registry = TriggerRegistry.create_default()

    text = (
        "Пожалуйста, загрузи навык [LOAD_SKILL: renpy_tags].\n"
        "Если требуется пауза: [PAUSE: Жду подтверждения пользователя].\n"
        "Также [DELEGATE: RhymeExpert | Подбери рифму к слову весна]."
    )

    calls = registry.parse_triggers(text)
    assert len(calls) == 3

    assert calls[0].name == "load_skill"
    assert calls[0].args == {"name": "renpy_tags"}

    assert calls[1].name == "pause"
    assert calls[1].args == {"reason": "Жду подтверждения пользователя"}

    assert calls[2].name == "delegate_task"
    assert calls[2].args == {"role": "RhymeExpert", "task": "Подбери рифму к слову весна"}


def test_trigger_registry_custom_trigger():
    """Проверка регистрации кастомного триггера с именованными группами."""
    registry = TriggerRegistry()
    registry.register(
        name="replace_word",
        pattern=r"\[REPLACE:\s*(?P<target>\w+)\s*->\s*(?P<replacement>\w+)\]",
        description="Заменить слово в тексте",
        example="[REPLACE: cat -> dog]",
    )

    text = "Нужно исправить [REPLACE: kot -> cat] в первой строке."
    calls = registry.parse_triggers(text)

    assert len(calls) == 1
    assert calls[0].name == "replace_word"
    assert calls[0].args == {"target": "kot", "replacement": "cat"}


def test_trigger_registry_strip_triggers():
    """Проверка очистки маркеров команд из текста ответа."""
    registry = TriggerRegistry.create_default()

    text = (
        "Начало ответа.\n"
        "[LOAD_SKILL: renpy_tags]\n"
        "Основной текст без маркеров. [PAUSE: Некоторая причина]\n"
        "Конец ответа."
    )

    cleaned = registry.strip_triggers(text)
    assert "[LOAD_SKILL" not in cleaned
    assert "[PAUSE" not in cleaned
    assert "Начало ответа." in cleaned
    assert "Основной текст без маркеров." in cleaned
    assert "Конец ответа." in cleaned


def test_trigger_registry_format_instructions():
    """Проверка генерации подсказки по текстовым командам для промпта."""
    registry = TriggerRegistry.create_default()
    instructions = registry.format_instructions()

    assert "--- FALLBACK TEXT COMMANDS ---" in instructions
    assert "[LOAD_SKILL: <name>]" in instructions
    assert "[PAUSE: <reason>]" in instructions
    assert "[DELEGATE: <role> | <task>]" in instructions


def test_prompt_builder_with_triggers():
    """Проверка добавления инструкций триггеров через PromptBuilder."""
    registry = TriggerRegistry.create_default()
    builder = PromptBuilder(template="System instruction.")

    prompt = builder.build(
        input_data={"test": "data"},
        triggers=registry,
    )

    assert "--- FALLBACK TEXT COMMANDS ---" in prompt
    assert "[LOAD_SKILL: <name>]" in prompt


def test_parser_parse_tool_calls_with_deduplication():
    """Проверка парсинга RobustLLMParser с дедупликацией вызовов из JSON и текстовых маркеров."""
    parser = RobustLLMParser()
    registry = TriggerRegistry.create_default()

    # Текст содержит ОДИНАКОВЫЙ вызов в формате JSON и в формате текстового маркера
    text = (
        '{"name": "load_skill", "args": {"name": "renpy_tags"}}\n'
        "Также дублирую маркер: [LOAD_SKILL: renpy_tags]\n"
        "И другой маркер: [PAUSE: Wait for input]"
    )

    calls = parser.parse_tool_calls(text, triggers=registry)

    # Должно быть только 2 вызова, а не 3 (load_skill дедуплицирован)
    assert len(calls) == 2
    assert calls[0].name == "load_skill"
    assert calls[0].args == {"name": "renpy_tags"}

    assert calls[1].name == "pause"
    assert calls[1].args == {"reason": "Wait for input"}


@pytest.mark.asyncio
async def test_agent_orchestrator_with_text_triggers(mock_provider):
    """Интеграционный тест: AgentOrchestrator обрабатывает текстовые триггеры и очищает их в финале."""
    custom_triggers = TriggerRegistry()
    custom_triggers.register(
        name="quick_calc",
        pattern=r"\[CALC:\s*(?P<expr>.*?)\]",
    )

    async def quick_calc(expr: str) -> str:
        return f"Calculated {expr}: 42"

    # 1. Модель выводит текстовый маркер вместо JSON
    # 2. Модель дает финальный ответ
    mock_provider.execute.side_effect = [
        "Мне нужно посчитать [CALC: 20 + 22]",
        "Готово! Результат получен: 42.",
    ]

    orchestrator = AgentOrchestrator(mock_provider)
    result = await orchestrator.execute_loop(
        "Посчитай выражение",
        tools={"quick_calc": quick_calc},
        triggers=custom_triggers,
    )

    assert result == "Готово! Результат получен: 42."
    assert mock_provider.execute.call_count == 2

    # Проверяем, что результат выполнения триггера попал во второй промпт
    second_prompt = mock_provider.execute.call_args_list[1][0][0]
    assert "Result of quick_calc: Calculated 20 + 22: 42" in second_prompt


@pytest.mark.asyncio
async def test_agent_orchestrator_strips_triggers_on_final_result(mock_provider):
    """Проверка очистки маркеров триггеров из финального ответа при завершении итераций."""
    custom_triggers = TriggerRegistry()
    custom_triggers.register(
        name="quick_calc",
        pattern=r"\[CALC:\s*(?P<expr>.*?)\]",
    )

    # Модель возвращает текст с маркером на единственной итерации
    mock_provider.execute.return_value = "Текст с маркером [CALC: 10 + 10] в конце."

    orchestrator = AgentOrchestrator(mock_provider)
    result = await orchestrator.execute_loop(
        "Запрос",
        tools={"quick_calc": AsyncMock(return_value="20")},
        triggers=custom_triggers,
        max_iterations=1,
    )

    assert "[CALC:" not in result
    assert "Текст с маркером" in result
    assert "в конце." in result
