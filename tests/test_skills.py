"""Тесты для подсистемы динамических навыков (Lazy Skills System)."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from chllm.builder import PromptBuilder
from chllm.exceptions import SkillValidationError
from chllm.orchestrator import AgentOrchestrator
from chllm.skills import (
    FileSystemSkillSource,
    InMemorySkillSource,
    SkillDefinition,
    SkillMetadata,
    SkillRegistry,
    SkillValidator,
)


@pytest.fixture
def sample_skill() -> SkillDefinition:
    """Фикстура тестового навыка в памяти."""
    return SkillDefinition(
        metadata=SkillMetadata(
            name="renpy_tags",
            description="Rules for RenPy formatting and tags",
            when_to_use="When translating strings with {i}, {b}, {color} or [variable] tags",
        ),
        instruction="Always preserve tags in square brackets and curly braces as-is.",
        tools={"validate_tags": lambda text: "{" in text},
        resources={"allowed_tags": ["i", "b", "u", "color"]},
    )


@pytest.fixture
def mock_provider():
    """Фикстура мок-провайдера LLM."""
    provider = MagicMock()
    provider.execute = AsyncMock()
    provider.pack_single_prompt = lambda p: p
    return provider


# --- 1. Тесты валидации и доменных моделей ---


def test_skill_metadata_validation():
    """Проверка валидации метаданных навыка."""
    valid = SkillMetadata(name="good_name-1", description="desc", when_to_use="when")
    assert valid.name == "good_name-1"

    # Некорректное имя (пробелы, спецсимволы)
    with pytest.raises(SkillValidationError, match="Недопустимое имя навыка"):
        SkillValidator.validate_name("bad name with spaces")

    with pytest.raises(SkillValidationError, match="Недопустимое имя навыка"):
        SkillValidator.validate_name("bad$name!")

    # Пустое описание
    with pytest.raises(SkillValidationError, match="непустое описание"):
        SkillValidator.validate_metadata(SkillMetadata(name="test", description="", when_to_use="when"))

    # Пустое when_to_use
    with pytest.raises(SkillValidationError, match="условия активации"):
        SkillValidator.validate_metadata(SkillMetadata(name="test", description="desc", when_to_use="   "))


def test_skill_definition_validation(sample_skill: SkillDefinition):
    """Проверка валидации полного определения навыка."""
    SkillValidator.validate_definition(sample_skill)

    # Пустая инструкция
    with pytest.raises(SkillValidationError, match="непустую инструкцию"):
        SkillValidator.validate_definition(
            SkillDefinition(
                metadata=sample_skill.metadata,
                instruction="   ",
            )
        )


def test_markdown_frontmatter_parsing():
    """Проверка парсинга markdown-файла со frontmatter."""
    valid_md = """---
name: dialogue_style
description: Dialog style guide
when_to_use: When characters speak informally
---
# Rules
Use friendly tone.
"""
    meta, body = SkillValidator.parse_and_validate_markdown(valid_md)
    assert meta.name == "dialogue_style"
    assert meta.description == "Dialog style guide"
    assert "Use friendly tone." in body

    # Отсутствие frontmatter
    with pytest.raises(SkillValidationError, match="должен начинаться с YAML Frontmatter"):
        SkillValidator.parse_and_validate_markdown("# Just markdown")

    # Битый YAML
    broken_yaml = """---
name: [unclosed
---
Body
"""
    with pytest.raises(SkillValidationError, match="Ошибка синтаксиса YAML"):
        SkillValidator.parse_and_validate_markdown(broken_yaml)


# --- 2. Тесты источников навыков ---


@pytest.mark.asyncio
async def test_in_memory_skill_source(sample_skill: SkillDefinition):
    """Проверка InMemorySkillSource."""
    source = InMemorySkillSource([sample_skill])

    catalog = await source.get_catalog()
    assert len(catalog) == 1
    assert catalog[0].name == "renpy_tags"

    loaded = await source.load_skill("renpy_tags")
    assert loaded is not None
    assert loaded.metadata.name == "renpy_tags"
    assert "validate_tags" in loaded.tools
    assert loaded.tools["validate_tags"]("{i}hello{/i}") is True

    # Несуществующий навык
    missing = await source.load_skill("unknown")
    assert missing is None


def test_filesystem_skill_source(tmp_path: Path):
    """Проверка загрузки структуры навыка из файловой системы."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # 1. Создаем папку скилла
    declension_dir = skills_dir / "declension"
    declension_dir.mkdir()

    (declension_dir / "SKILL.md").write_text(
        """---
name: declension
description: Helper for Russian name declensions
when_to_use: When translating names in accusative or dative case
---
Always check gender before declining.
""",
        encoding="utf-8",
    )

    (declension_dir / "tools.py").write_text(
        """
def decline_name(name: str, case: str) -> str:
    return f"{name}_{case}"

TOOLS = {"decline_name": decline_name}
""",
        encoding="utf-8",
    )

    res_dir = declension_dir / "resources"
    res_dir.mkdir()
    (res_dir / "exceptions.json").write_text(json.dumps({"Ivan": "Ivany"}), encoding="utf-8")

    # 2. Создаем плоский файл скилла
    (skills_dir / "simple.md").write_text(
        """---
name: simple_note
description: Simple guideline
when_to_use: For general instructions
---
Be concise.
""",
        encoding="utf-8",
    )

    # Инициализируем источник
    source = FileSystemSkillSource(skills_dir, validate_on_init=True)
    catalog = asyncio.run(source.get_catalog())
    assert len(catalog) == 2

    # Загружаем скилл с инструментами и ресурсами
    skill = asyncio.run(source.load_skill("declension"))
    assert skill is not None
    assert skill.metadata.name == "declension"
    assert "decline_name" in skill.tools
    assert skill.tools["decline_name"]("Max", "gen") == "Max_gen"
    assert "exceptions.json" in skill.resources
    assert skill.resources["exceptions.json"] == {"Ivan": "Ivany"}


def test_filesystem_skill_source_fail_fast(tmp_path: Path):
    """Проверка Fail-Fast валидации при обнаружении битого синтаксиса в tools.py."""
    skills_dir = tmp_path / "broken_skills"
    skills_dir.mkdir()

    bad_dir = skills_dir / "bad_skill"
    bad_dir.mkdir()

    (bad_dir / "SKILL.md").write_text(
        """---
name: bad_skill
description: Bad skill
when_to_use: Always
---
Instruction
""",
        encoding="utf-8",
    )

    # Синтаксическая ошибка в tools.py
    (bad_dir / "tools.py").write_text("def broken_syntax(:", encoding="utf-8")

    with pytest.raises(SkillValidationError, match="Синтаксическая ошибка в коде инструментов"):
        FileSystemSkillSource(skills_dir, validate_on_init=True)


# --- 3. Тесты реестра и PromptBuilder ---


def test_prompt_builder_with_skills(sample_skill: SkillDefinition):
    """Проверка внедрения каталога навыков в PromptBuilder."""
    registry = SkillRegistry()
    registry.register_skill(sample_skill)

    builder = PromptBuilder(template="Translate the text.")
    prompt = builder.build(
        input_data={"text": "Hello {player}"},
        skills=registry,
    )

    assert "--- AVAILABLE SKILLS ---" in prompt
    assert "renpy_tags" in prompt
    assert "When translating strings with {i}" in prompt
    assert "load_skill(name='...')" in prompt


# --- 4. Тесты AgentOrchestrator с динамическими навыками ---


@pytest.mark.asyncio
async def test_agent_orchestrator_loads_skill_via_tool_call(mock_provider, sample_skill: SkillDefinition):
    """Тест загрузки навыка через стандартный tool call load_skill и динамического вызова его функций."""
    registry = SkillRegistry()
    registry.register_skill(sample_skill)

    # Сценарий:
    # 1. LLM видит, что нужен скилл и вызывает `load_skill`
    # 2. На следующей итерации LLM видит новые инструменты и вызывает `validate_tags`
    # 3. Финальный ответ
    mock_provider.execute.side_effect = [
        '{"name": "load_skill", "args": {"name": "renpy_tags"}}',
        '{"name": "validate_tags", "args": {"text": "{i}Привет{/i}"}}',
        "Перевод выполнен успешно.",
    ]

    orchestrator = AgentOrchestrator(mock_provider)
    result = await orchestrator.execute_loop("Переведи: {i}Hello{/i}", skills=registry)

    assert result == "Перевод выполнен успешно."
    assert mock_provider.execute.call_count == 3

    # Проверяем, что в промпте 2-й итерации появилась системная инструкция загруженного скилла
    second_call_prompt = mock_provider.execute.call_args_list[1][0][0]
    assert "--- ACTIVE SKILLS INSTRUCTIONS ---" in second_call_prompt
    assert "[SKILL: renpy_tags]" in second_call_prompt
    assert "Always preserve tags" in second_call_prompt


@pytest.mark.asyncio
async def test_agent_orchestrator_loads_skill_via_fallback_token(mock_provider, sample_skill: SkillDefinition):
    """Тест загрузки навыка через текстовый fallback маркер [LOAD_SKILL: <name>]."""
    registry = SkillRegistry()
    registry.register_skill(sample_skill)

    # 1. Модель выводит маркер без tool calling
    # 2. Модель дает финальный ответ
    mock_provider.execute.side_effect = [
        "Мне нужны правила по тегам [LOAD_SKILL: renpy_tags]",
        "Готово, теги сохранены.",
    ]

    orchestrator = AgentOrchestrator(mock_provider)
    result = await orchestrator.execute_loop("Переведи строку", skills=registry)

    assert result == "Готово, теги сохранены."
    assert mock_provider.execute.call_count == 2

    # Во втором запросе инструкция скилла должна присутствовать
    second_call_prompt = mock_provider.execute.call_args_list[1][0][0]
    assert "[SKILL: renpy_tags]" in second_call_prompt
    assert "Result of load_skill: Skill 'renpy_tags' successfully loaded" in second_call_prompt


@pytest.mark.asyncio
async def test_agent_orchestrator_preloaded_skills(mock_provider, sample_skill: SkillDefinition):
    """Тест предварительной загрузки (preload_skills) с Zero-RTT."""
    registry = SkillRegistry()
    registry.register_skill(sample_skill)

    # LLM дает ответ с 1-й же итерации, так как скилл уже загружен заранее
    mock_provider.execute.side_effect = [
        "Финальный перевод с тегами.",
    ]

    orchestrator = AgentOrchestrator(mock_provider)
    result = await orchestrator.execute_loop(
        "Переведи строку",
        skills=registry,
        preload_skills=["renpy_tags"],
    )

    assert result == "Финальный перевод с тегами."
    assert mock_provider.execute.call_count == 1

    first_call_prompt = mock_provider.execute.call_args_list[0][0][0]
    assert "--- ACTIVE SKILLS INSTRUCTIONS ---" in first_call_prompt
    assert "[SKILL: renpy_tags]" in first_call_prompt


@pytest.mark.asyncio
async def test_agent_orchestrator_unknown_skill_graceful_error(mock_provider, sample_skill: SkillDefinition):
    """Тест мягкой обработки запроса неизвестного навыка."""
    registry = SkillRegistry()
    registry.register_skill(sample_skill)

    mock_provider.execute.side_effect = [
        '{"name": "load_skill", "args": {"name": "non_existent"}}',
        "Понял, работаю без этого навыка.",
    ]

    orchestrator = AgentOrchestrator(mock_provider)
    result = await orchestrator.execute_loop("Запрос", skills=registry)

    assert result == "Понял, работаю без этого навыка."
    second_call_prompt = mock_provider.execute.call_args_list[1][0][0]
    assert "Error: Skill 'non_existent' not found. Available skills: renpy_tags" in second_call_prompt
