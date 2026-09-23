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


class CustomCard(BaseModel):
    title: str
    points: int


def test_parse_explicit_container_key(parser):
    """Тест парсинга с явным указанием container_key."""
    raw = '{"cards": [{"title": "Dragon", "points": 10}, {"title": "Goblin", "points": 2}]}'
    result = parser.parse(raw, validation_model=CustomCard, container_key="cards")

    assert "cards" in result
    assert len(result["cards"]) == 2
    assert isinstance(result["cards"][0], CustomCard)
    assert result["cards"][0].title == "Dragon"
    assert result["cards"][0].points == 10


def test_parse_container_keys_tuple(parser):
    """Тест парсинга с указанием кортежа container_keys."""
    raw = '{"dialogues": [{"title": "Hello", "points": 1}]}'
    result = parser.parse(raw, validation_model=CustomCard, container_keys=("items", "dialogues"))

    # По умолчанию возвращается ключ 'batch', если container_key не задан
    assert "batch" in result
    assert len(result["batch"]) == 1
    assert result["batch"][0].title == "Hello"


def test_parse_zero_config_fallback_autodetect(parser):
    """Тест Zero-Config fallback: автоматический поиск списка по сигнатуре Pydantic-модели."""
    # LLM вернула неожиданный ключ 'unpredicted_loot_box'
    raw = '{"status": "ok", "unpredicted_loot_box": [{"title": "Sword", "points": 100}]}'
    result = parser.parse(raw, validation_model=CustomCard)

    assert len(result["batch"]) == 1
    assert isinstance(result["batch"][0], CustomCard)
    assert result["batch"][0].title == "Sword"
    assert result["batch"][0].points == 100


def test_parse_custom_field_aliases(parser):
    """Тест нормализации полей через field_aliases."""

    class Person(BaseModel):
        full_name: str
        age: int

    raw = '{"items": [{"name": "Alice", "age": 30}]}'
    result = parser.parse(
        raw,
        validation_model=Person,
        container_key="items",
        field_aliases={"name": "full_name"},
    )
    assert len(result["items"]) == 1
    assert result["items"][0].full_name == "Alice"
    assert result["items"][0].age == 30


def test_parse_items_convenience_method(parser):
    """Тест метода parse_items, возвращающего сразу список объектов."""
    raw = '{"cards": [{"title": "Knight", "points": 5}]}'
    cards = parser.parse_items(raw, validation_model=CustomCard, container_key="cards")

    assert isinstance(cards, list)
    assert len(cards) == 1
    assert isinstance(cards[0], CustomCard)
    assert cards[0].title == "Knight"
    assert cards[0].points == 5


def test_parse_include_empty_glossary_flag(parser):
    """Тест отключения пустого глоссария в выводе."""
    raw = '[{"id": "1", "translated_text": "Привет"}]'
    result = parser.parse(raw, include_empty_glossary=False)
    assert "suggested_glossary_terms" not in result
    assert "batch" in result
