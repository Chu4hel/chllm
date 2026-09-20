from chllm.context import ContextBuilder


def test_chain_context_basic():
    # Набор данных для обработки
    items = ["Line 1", "Line 2", "Line 3"]
    # История из предыдущих батчей
    history = ["Prev 1", "Prev 2"]

    builder = ContextBuilder(strategy="chain")

    # Мы передаем элементы и функцию для форматирования элемента в строку контекста
    result = builder.build(items, initial_history=history)

    # Ожидаемое поведение:
    # Элемент 0: получает всю начальную историю
    # Элемент 1: получает только Элемент 0
    # Элемент 2: получает только Элемент 1

    assert result[0].context == ["Prev 1", "Prev 2"]
    assert result[0].data == "Line 1"

    assert result[1].context == ["Line 1"]
    assert result[2].context == ["Line 2"]


def test_chain_context_with_formatter():
    items = [{"text": "Hello", "id": 1}, {"text": "World", "id": 2}]

    # Кастомный форматтер для сложных объектов
    def my_formatter(item):
        return f"MSG: {item['text']}"

    builder = ContextBuilder(strategy="chain")
    result = builder.build(items, formatter=my_formatter)

    assert result[0].context == []
    assert result[1].context == ["MSG: Hello"]
