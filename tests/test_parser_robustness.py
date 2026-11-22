import pytest
from pydantic import BaseModel

from chllm.parser import RobustLLMParser


@pytest.fixture
def parser():
    return RobustLLMParser()


def test_recovery_nested_json(parser):
    # Сложный вложенный JSON, оборванный в конце
    raw = '{"batch": [{"id": "1", "data": {"inner": "val"'  # Оборвано здесь
    result = parser.parse(raw)

    # Мы ожидаем, что парсер закроет и внутренний объект, и список, и внешний объект
    assert len(result["batch"]) == 1
    assert result["batch"][0]["id"] == "1"
    assert result["batch"][0]["data"]["inner"] == "val"


def test_recovery_array_cut(parser):
    # Массив объектов, оборванный на запятой или в середине объекта
    raw = '[{"id": "1"}, {"id": "2"'  # Оборвано
    result = parser.parse(raw)

    # Должен спасти хотя бы первый объект, если второй совсем битый
    assert len(result["batch"]) >= 1
    assert result["batch"][0]["id"] == "1"


def test_recovery_incomplete_string(parser):
    # Оборвано внутри строки
    raw = '{"id": "1", "text": "This is very lon'  # Оборвано без кавычки
    result = parser.parse(raw)

    assert len(result["batch"]) == 1
    assert result["batch"][0]["id"] == "1"
    # Текст может быть обрезан, но JSON должен быть валиден.
    # Парсер нормализует 'text' в 'translated_text'
    assert "translated_text" in result["batch"][0]
    assert result["batch"][0]["translated_text"] == "This is very lon"


class SampleWord(BaseModel):
    word: str
    nom: str | None = None
    gen: str | None = None


def test_sanitize_conversational_text(parser):
    raw = """
    Конечно! Вот ваш JSON-ответ:
    ```json
    {
      «word»: «кошка»,
      «nom»: «кошка»,
      «gen»: «кошки»,
    }
    ```
    Всегда рад помочь!
    """
    sanitized = parser.sanitize_conversational_text(raw)
    assert "Конечно" not in sanitized
    assert "Всегда рад" not in sanitized
    assert "«" not in sanitized
    assert '"word": "кошка"' in sanitized
    assert '"gen": "кошки"' in sanitized


def test_parse_conversational_response_with_model(parser):
    raw = """
    Вот предложенные варианты склонений:
    ```json
    {
      "declensions": [
        {
          «word»: «кот»,
          «nom»: «кот»,
          «gen»: «кота»,
        },
        {
          «word»: «собака»,
          «nom»: «собака»,
          «gen»: «собаки»,
        }
      ]
    }
    ```
    Надеюсь, перевод точный!
    """
    items = parser.parse_conversational_response(raw, validation_model=SampleWord)
    assert len(items) == 2
    assert isinstance(items[0], SampleWord)
    assert items[0].word == "кот"
    assert items[0].gen == "кота"
    assert items[1].word == "собака"
    assert items[1].gen == "собаки"


def test_parse_conversational_unbracketed_comma_separated_objects(parser):
    raw = """
    Конечно, держи:
    {"word": "стол", "nom": "стол", "gen": "стола"},
    {"word": "стул", "nom": "стул", "gen": "стула"},
    Удачи в переводе!
    """
    items = parser.parse_conversational_response(raw, validation_model=SampleWord)
    assert len(items) == 2
    assert items[0].word == "стол"
    assert items[1].word == "стул"


def test_parse_conversational_single_quotes(parser):
    raw = """
    Here is the result:
    {'declensions': [{'word': 'дом', 'nom': 'дом', 'gen': 'дома'}]}
    """
    items = parser.parse_conversational_response(raw, validation_model=SampleWord)
    assert len(items) == 1
    assert items[0].word == "дом"
    assert items[0].gen == "дома"


def test_parse_conversational_without_model(parser):
    raw = """
    Вот список:
    [
      {"original": "Game", "localized": "Игра",},
      {"original": "Settings", "localized": "Настройки",},
    ]
    """
    items = parser.parse_conversational_response(raw)
    assert isinstance(items, list)
    assert len(items) == 2
    assert items[0]["localized"] == "Игра"
    assert items[1]["localized"] == "Настройки"


def test_parse_conversational_empty_or_whitespace(parser):
    assert parser.parse_conversational_response("") == []
    assert parser.parse_conversational_response("   \n\t  ") == []


def test_parse_conversational_regex_fallback(parser):
    raw = """
    Часть 1:
    {"word": "лес", "nom": "лес", "gen": "леса"}
    Какой-то промежуточный невалидный текст или болтовня
    Часть 2:
    {"word": "поле", "nom": "поле", "gen": "поля"}
    """
    items = parser.parse_conversational_response(raw, validation_model=SampleWord)
    assert len(items) == 2
    assert items[0].word == "лес"
    assert items[1].word == "поле"


def test_parse_raw_with_typographic_quotes_and_trailing_commas(parser):
    raw = """
    {
      «title»: «Игра»,
      «version»: 1,
    }
    """
    data = parser.parse_raw(raw)
    assert isinstance(data, dict)
    assert data["title"] == "Игра"
    assert data["version"] == 1
