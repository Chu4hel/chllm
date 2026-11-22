"""Тесты для оболочки вызова внешних CLI-инструментов (CliTool)."""

import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from chllm.exceptions import CliNotFoundError, CliProcessError, CliTimeoutError
from chllm.orchestrator import AgentOrchestrator
from chllm.skills import SkillDefinition, SkillMetadata, SkillRegistry
from chllm.tools import CliInputMode, CliTool, CliToolConfig


@pytest.fixture
def mock_provider():
    """Фикстура мок-провайдера LLM."""
    provider = MagicMock()
    provider.execute = AsyncMock()
    provider.pack_single_prompt = lambda p: p
    return provider


@pytest.mark.asyncio
async def test_cli_tool_stdin_json_mode():
    """Проверка передачи аргументов через stdin в формате JSON."""
    script = (
        "import sys, json\n"
        "data = json.load(sys.stdin)\n"
        "print(f\"Hello, {data.get('name')}! Count: {data.get('count')}\")\n"
    )

    tool = CliTool(
        name="greeter",
        config=CliToolConfig(
            command=[sys.executable, "-c", script],
            input_mode=CliInputMode.STDIN_JSON,
        ),
    )

    result = await tool(name="RenPy", count=42)
    assert result == "Hello, RenPy! Count: 42"


@pytest.mark.asyncio
async def test_cli_tool_cli_args_mode():
    """Проверка передачи аргументов в виде CLI-флагов."""
    script = "import sys\nprint(' '.join(sys.argv[1:]))\n"

    tool = CliTool(
        name="arg_echo",
        config=CliToolConfig(
            command=[sys.executable, "-c", script],
            input_mode=CliInputMode.CLI_ARGS,
        ),
    )

    result = await tool(target="player", dry_run=True, count=5)
    assert "--target player" in result
    assert "--dry-run" in result
    assert "--count 5" in result


@pytest.mark.asyncio
async def test_cli_tool_strip_ansi():
    """Проверка удаления ANSI escape-последовательностей из вывода."""
    script = "print('\\x1b[32m[SUCCESS]\\x1b[0m Translation complete')"

    tool = CliTool(
        name="ansi_test",
        config=CliToolConfig(
            command=[sys.executable, "-c", script],
            strip_ansi=True,
        ),
    )

    result = await tool()
    assert result == "[SUCCESS] Translation complete"
    assert "\x1b" not in result


@pytest.mark.asyncio
async def test_cli_tool_output_truncation():
    """Проверка ограничения максимальной длины вывода."""
    script = "print('A' * 200)"

    tool = CliTool(
        name="long_output",
        config=CliToolConfig(
            command=[sys.executable, "-c", script],
            max_output_length=50,
        ),
    )

    result = await tool()
    assert len(result) > 50  # включает суффикс усечения
    assert "[... Output truncated: 200 characters total ...]" in result
    assert result.startswith("A" * 50)


@pytest.mark.asyncio
async def test_cli_tool_parse_json_response():
    """Проверка извлечения валидного JSON из зашумленного вывода."""
    script = "print('Log banner line 1')\nprint('{\"status\": \"ok\", \"items\": [1, 2]}')\nprint('Debug log line 2')\n"

    tool = CliTool(
        name="json_emitter",
        config=CliToolConfig(
            command=[sys.executable, "-c", script],
            parse_json_response=True,
        ),
    )

    result = await tool()
    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    assert parsed["items"] == [1, 2]


@pytest.mark.asyncio
async def test_cli_tool_timeout_handling():
    """Проверка обработки таймаута: возврат мягкой ошибки vs исключение."""
    script = "import time; time.sleep(2)"

    # 1. Мягкая ошибка (по умолчанию)
    soft_tool = CliTool(
        name="slow_process",
        config=CliToolConfig(
            command=[sys.executable, "-c", script],
            timeout=0.2,
            raise_on_error=False,
        ),
    )
    result = await soft_tool()
    assert "Tool Execution Failed: Process timed out after 0.2s" in result

    # 2. Бросок исключения
    strict_tool = CliTool(
        name="slow_process_strict",
        config=CliToolConfig(
            command=[sys.executable, "-c", script],
            timeout=0.2,
            raise_on_error=True,
        ),
    )
    with pytest.raises(CliTimeoutError, match="Process timed out"):
        await strict_tool()


@pytest.mark.asyncio
async def test_cli_tool_exit_code_handling():
    """Проверка обработки ненулевого кода возврата."""
    script = "import sys\nsys.stderr.write('Fatal memory error!\\n')\nsys.exit(42)\n"

    # 1. Мягкая ошибка
    soft_tool = CliTool(
        name="crasher",
        config=CliToolConfig(
            command=[sys.executable, "-c", script],
            raise_on_error=False,
        ),
    )
    result = await soft_tool()
    assert "Tool Execution Failed (exit code 42): Fatal memory error!" in result

    # 2. Бросок исключения
    strict_tool = CliTool(
        name="crasher_strict",
        config=CliToolConfig(
            command=[sys.executable, "-c", script],
            raise_on_error=True,
        ),
    )
    with pytest.raises(CliProcessError) as exc_info:
        await strict_tool()
    assert exc_info.value.returncode == 42
    assert "Fatal memory error!" in exc_info.value.stderr


@pytest.mark.asyncio
async def test_cli_tool_not_found():
    """Проверка реакции на несуществующую команду."""
    soft_tool = CliTool(
        name="missing",
        config=CliToolConfig(
            command=["non_existent_binary_xyz_123"],
            raise_on_error=False,
        ),
    )
    result = await soft_tool()
    assert "Tool Execution Failed: Command 'non_existent_binary_xyz_123' not found" in result

    strict_tool = CliTool(
        name="missing_strict",
        config=CliToolConfig(
            command=["non_existent_binary_xyz_123"],
            raise_on_error=True,
        ),
    )
    with pytest.raises(CliNotFoundError, match="not found"):
        await strict_tool()


@pytest.mark.asyncio
async def test_cli_tool_orchestrator_integration(mock_provider):
    """Интеграционный тест CliTool с AgentOrchestrator."""
    script = "import sys, json\ndata = json.load(sys.stdin)\nprint(f\"Extracted: {data.get('term')}\")\n"

    cli_tool = CliTool(
        name="lookup_term",
        config=CliToolConfig(command=[sys.executable, "-c", script]),
    )

    # 1. LLM просит вызвать lookup_term
    # 2. LLM дает финальный ответ
    mock_provider.execute.side_effect = [
        '{"name": "lookup_term", "args": {"term": "Eileen"}}',
        "Термин Eileen найден в глоссарии.",
    ]

    orchestrator = AgentOrchestrator(mock_provider)
    result = await orchestrator.execute_loop("Найди Eileen", tools={"lookup_term": cli_tool})

    assert result == "Термин Eileen найден в глоссарии."
    second_prompt = mock_provider.execute.call_args_list[1][0][0]
    assert "Result of lookup_term: Extracted: Eileen" in second_prompt


@pytest.mark.asyncio
async def test_cli_tool_inside_dynamic_skill(mock_provider):
    """Тест использования CliTool внутри SkillDefinition в Lazy Skills System."""
    script = "import sys, json\ndata = json.load(sys.stdin)\nprint(f\"RPYC decompiled: {data.get('file')}\")\n"

    decompiler_tool = CliTool(
        name="decompile_rpyc",
        config=CliToolConfig(command=[sys.executable, "-c", script]),
    )

    skill = SkillDefinition(
        metadata=SkillMetadata(
            name="renpy_decompiler",
            description="Decompiles bytecode",
            when_to_use="When reading .rpyc files",
        ),
        instruction="Use decompile_rpyc tool to inspect .rpyc files.",
        tools={"decompile_rpyc": decompiler_tool},
    )

    registry = SkillRegistry()
    registry.register_skill(skill)

    mock_provider.execute.side_effect = [
        '{"name": "load_skill", "args": {"name": "renpy_decompiler"}}',
        '{"name": "decompile_rpyc", "args": {"file": "script.rpyc"}}',
        "Файл декомпилирован успешно.",
    ]

    orchestrator = AgentOrchestrator(mock_provider)
    result = await orchestrator.execute_loop("Декомпилируй script.rpyc", skills=registry)

    assert result == "Файл декомпилирован успешно."
    third_prompt = mock_provider.execute.call_args_list[2][0][0]
    assert "Result of decompile_rpyc: RPYC decompiled: script.rpyc" in third_prompt
