import pytest

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
