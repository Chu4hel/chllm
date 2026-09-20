"""
Пример использования RobustLLMParser для восстановления поврежденного JSON.
"""

from pydantic import BaseModel

from chllm import RobustLLMParser


class Product(BaseModel):
    """Модель данных товара для демонстрации парсинга."""

    name: str
    price: float


def run_demo() -> None:
    """Запускает демонстрацию восстановления поврежденного JSON ответа."""
    parser = RobustLLMParser()

    # Пример 1: ИИ обрезал ответ на середине
    raw_response = """
    Вот список товаров в формате JSON:
    ```json
    [
      {"name": "Laptop", "price": 999.99},
      {"name": "Smartphone", "price": 49
    """  # JSON не закрыт, кавычка оборвана

    print("--- Исходный текст ---")
    print(raw_response)

    result = parser.parse(raw_response, validation_model=Product)

    print("\n--- Результат парсинга (бач) ---")
    for item in result["batch"]:
        if isinstance(item, Product):
            print(f"Товар: {item.name}, Цена: {item.price}")
        else:
            print(f"Товар: {item.get('name')}, Цена: {item.get('price')}")

    # Пример 2: Глубокая вложенность
    nested_raw = '{"data": {"user": {"id": 1, "meta": {"bio": "Hello'
    raw_obj = parser.parse_raw(nested_raw)

    print("\n--- Восстановление вложенного объекта ---")
    print(f"Полученный объект: {raw_obj}")


if __name__ == "__main__":
    run_demo()
