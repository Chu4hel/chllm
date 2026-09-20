import pytest
from pydantic import BaseModel

from chllm.parser import RobustLLMParser


class MockItem(BaseModel):
    id: str
    translated_text: str


class MockTerm(BaseModel):
    term: str
    translation: str


@pytest.fixture
def parser():
    return RobustLLMParser()


def test_parse_simple_json_array(parser):
    """Тест парсинга простого JSON массива в словари."""
    raw = '[{"id": "1", "translated_text": "Привет"}]'
    result = parser.parse(raw)
    assert len(result["batch"]) == 1
    assert result["batch"][0]["translated_text"] == "Привет"


def test_parse_with_pydantic_model(parser):
    """Тест парсинга с автоматической валидацией через Pydantic."""
    raw = '{"batch": [{"id": "1", "translated_text": "Привет"}]}'
    result = parser.parse(raw, validation_model=MockItem)
    assert len(result["batch"]) == 1
    assert isinstance(result["batch"][0], MockItem)
    assert result["batch"][0].translated_text == "Привет"


def test_parse_json_lines_mixed(parser):
    """Тест парсинга JSON Lines с глоссарием."""
    raw = """
    {"id": "1", "translated_text": "Раз"}
    {"id": "2", "translated_text": "Два"}
    {"suggested_glossary_terms": [{"term": "Word", "translation": "Слово"}]}
    """
    result = parser.parse(raw, validation_model=MockItem, glossary_model=MockTerm)
    assert len(result["batch"]) == 2
    assert len(result["suggested_glossary_terms"]) == 1
    assert isinstance(result["suggested_glossary_terms"][0], MockTerm)
    assert result["suggested_glossary_terms"][0].term == "Word"


def test_parse_markdown_cleaning(parser):
    """Тест очистки Markdown оберток."""
    raw = "Вот ваш ответ:\n```json\n" + '[{"id": "1", "translated_text": "Тест"}]' + "\n```\nНадеюсь помог!"
    result = parser.parse(raw)
    assert len(result["batch"]) == 1
    assert result["batch"][0]["translated_text"] == "Тест"


def test_parse_recovery_unclosed_brace(parser):
    """Тест восстановления оборванного JSON (незакрытая скобка)."""
    raw = '{"id": "1", "translated_text": "Текст"'  # Пропущена }
    result = parser.parse(raw)
    assert len(result["batch"]) == 1
    assert result["batch"][0]["translated_text"] == "Текст"


def test_parse_validation_error_logging(parser, mocker):
    """Тест логирования ошибки при неверном формате данных для модели."""
    mock_logger = mocker.Mock()
    p = RobustLLMParser(logger=mock_logger)

    # Используем глоссарий, так как там проверка менее строгая перед вызовом _add_item
    # Передаем термин без обязательного поля 'translation'
    raw = '{"suggested_glossary_terms": [{"term": "MissingTranslation"}]}'

    result = p.parse(raw, glossary_model=MockTerm)
    assert len(result["suggested_glossary_terms"]) == 0
    mock_logger.warning.assert_called()
