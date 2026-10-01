import pytest
from pydantic import BaseModel

from chllm.builder import PromptBuilder


class MockData(BaseModel):
    id: int
    text: str


@pytest.fixture
def builder():
    return PromptBuilder(template="Основная инструкция")


def test_builder_simple_build(builder):
    """Тест базовой сборки промпта."""
    data = {"key": "value"}
    result = builder.build(input_data=data)

    assert "Основная инструкция" in result
    assert "--- INPUT DATA ---" in result
    assert '"key": "value"' in result


def test_builder_with_context(builder):
    """Тест сборки с контекстом."""
    context = ["Предыдущая фраза 1", "Предыдущая фраза 2"]
    result = builder.build(input_data=[], context=context)

    assert "--- CONTEXT ---" in result
    assert "Предыдущая фраза 1" in result
    assert "Предыдущая фраза 2" in result


def test_builder_additional_instruction(builder):
    """Тест добавления системной инструкции."""
    result = builder.build(input_data={}, instruction="Новая инструкция")

    assert "Новая инструкция" in result
    assert "Основная инструкция" in result


def test_builder_with_pydantic_model(builder):
    """Тест сериализации Pydantic модели."""
    data = MockData(id=1, text="Привет")
    result = builder.build(input_data=data)

    assert '"id": 1' in result
    assert '"text": "Привет"' in result


def test_builder_custom_label(builder):
    """Тест использования кастомного заголовка данных."""
    result = builder.build(input_data={}, data_label="MY DATA")
    assert "--- MY DATA ---" in result


def test_builder_with_mock_input():
    """Проверка, что строитель не падает, если данные - это Mock (актуально для тестов)."""
    from unittest.mock import MagicMock

    builder = PromptBuilder()
    mock_data = MagicMock()
    # Имитируем, что модель возвращает другой мок при вызове model_dump_json
    mock_data.model_dump_json.return_value = MagicMock()

    prompt = builder.build(mock_data)
    assert "INPUT DATA" in prompt
    # Проверяем, что мок преобразовался в строку (представление мока)
    assert "MagicMock" in prompt


def test_builder_custom_context_label(builder):
    """Тест использования кастомного заголовка контекста."""
    result = builder.build(input_data={}, context=["История сообщений"], context_label="CHAT HISTORY")
    assert "--- CHAT HISTORY ---" in result
    assert "История сообщений" in result


def test_builder_with_response_model(builder):
    """Тест автоматического добавления схемы модели в промпт."""

    class TargetOutput(BaseModel):
        summary: str
        score: int

    prompt = builder.build(input_data={"raw": "text"}, response_model=TargetOutput)
    assert "ОТВЕТ ДОЛЖЕН БЫТЬ СТРОГО В ФОРМАТЕ JSON" in prompt
    assert '"summary"' in prompt
    assert '"score"' in prompt


def test_builder_add_few_shot_examples_tuples(builder):
    """Тест добавления Few-Shot примеров кортежами (вход, ответ)."""

    class OptionModel(BaseModel):
        title: str
        score: int

    examples = [
        ("Input String 1", {"title": "Title 1", "score": 10}),
        ({"raw": "Input Dict"}, OptionModel(title="Title 2", score=20)),
    ]

    builder.add_few_shot_examples(examples)
    prompt = builder.build(input_data={"current": "data"})

    assert "--- EXAMPLES ---" in prompt
    assert "Example 1:" in prompt
    assert "Input String 1" in prompt
    assert '"title": "Title 1"' in prompt
    assert "Example 2:" in prompt
    assert '"raw": "Input Dict"' in prompt
    assert '"title": "Title 2"' in prompt
    assert "--- INPUT DATA ---" in prompt


def test_builder_add_few_shot_with_description(builder):
    """Тест примеров с опциональным описанием."""
    examples = [
        ("Hero Name", {"word": "Саня"}, "Разговорная форма"),
    ]
    builder.add_few_shot_examples(examples)
    prompt = builder.build(input_data="Current Name")

    assert "Example 1 (Разговорная форма):" in prompt
    assert "Hero Name" in prompt
    assert '"word": "Саня"' in prompt


def test_builder_few_shot_direct_in_build(builder):
    """Тест передачи few_shot_examples прямо в метод build."""
    examples = [("X", {"y": 1})]
    prompt = builder.build(
        input_data="Test",
        few_shot_examples=examples,
        examples_label="ДЕМОНСТРАЦИИ",
    )

    assert "--- ДЕМОНСТРАЦИИ ---" in prompt
    assert "Example 1:" in prompt
    assert '"y": 1' in prompt


def test_builder_few_shot_chaining_and_clear(builder):
    """Тест цепочечных вызовов add_few_shot_example и clear_examples."""
    prompt = builder.add_few_shot_example("Q1", "A1").add_few_shot_example("Q2", "A2").build(input_data="Q3")
    assert "Example 1:" in prompt
    assert "Example 2:" in prompt

    builder.clear_examples()
    clean_prompt = builder.build(input_data="Q4")
    assert "--- EXAMPLES ---" not in clean_prompt
    assert "Example 1:" not in clean_prompt
