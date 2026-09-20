from chllm.parser import RobustLLMParser


def test_parse_tool_calls_valid_json():
    """Тест успешного извлечения одного инструмента из JSON."""
    parser = RobustLLMParser()
    raw_text = """
    Я выполню это через инструмент.
    ```json
    {
        "name": "get_git_commits",
        "args": {"count": 5}
    }
    ```
    """
    results = parser.parse_tool_calls(raw_text)
    assert len(results) == 1
    assert results[0].name == "get_git_commits"
    assert results[0].args == {"count": 5}


def test_parse_tool_calls_multiple():
    """Тест извлечения нескольких инструментов."""
    parser = RobustLLMParser()
    raw_text = """
    Инструмент 1: {"name": "tool1", "args": {}}
    Инструмент 2: {"name": "tool2", "args": {"x": 1}}
    """
    results = parser.parse_tool_calls(raw_text)
    assert len(results) == 2
    assert results[0].name == "tool1"
    assert results[1].name == "tool2"


def test_parse_tool_calls_invalid_structure():
    """Тест обработки JSON, который не является инструментом (нет name)."""
    parser = RobustLLMParser()
    raw_text = '{"some_other": "data"}'
    results = parser.parse_tool_calls(raw_text)
    assert len(results) == 0


def test_parse_tool_calls_nested_tool_use():
    """Тест извлечения инструмента, обернутого в tool_use и использующего parameters вместо args."""
    parser = RobustLLMParser()
    raw_text = """
    {
      "tool_use": {
        "name": "get_rat_metrics",
        "parameters": {
          "version_uuid": "93dba73f-17a5-4495-bd3f-9c460d695e52"
        }
      }
    }
    """
    results = parser.parse_tool_calls(raw_text)
    assert len(results) == 1
    assert results[0].name == "get_rat_metrics"
    assert results[0].args == {"version_uuid": "93dba73f-17a5-4495-bd3f-9c460d695e52"}


def test_parse_tool_calls_flat_parameters():
    """Тест извлечения инструмента с parameters вместо args на верхнем уровне."""
    parser = RobustLLMParser()
    raw_text = """
    {
      "name": "get_rat_metrics",
      "parameters": {
        "version_uuid": "v0.2.0-pc"
      }
    }
    """
    results = parser.parse_tool_calls(raw_text)
    assert len(results) == 1
    assert results[0].name == "get_rat_metrics"
    assert results[0].args == {"version_uuid": "v0.2.0-pc"}


def test_parse_tool_calls_flat_arguments():
    """Тест извлечения параметров без вложенности args/parameters."""
    parser = RobustLLMParser()
    raw_text = """
    {
      "name": "get_rat_metrics",
      "version_uuid": "v0.2.0-pc"
    }
    """
    results = parser.parse_tool_calls(raw_text)
    assert len(results) == 1
    assert results[0].name == "get_rat_metrics"
    assert results[0].args == {"version_uuid": "v0.2.0-pc"}
