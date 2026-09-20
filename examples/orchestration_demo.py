"""
Демонстрация работы Orchestrator: деление батча и ретраи.
"""

import asyncio

from chllm import ContentBlockedError, Orchestrator, RateLimitError, RetryStrategy


class FakeAIProvider:
    """Мок-провайдер, имитирующий проблемы с лимитами и цензурой."""

    def __init__(self) -> None:
        """Инициализирует тестовый провайдер."""
        self.attempts: int = 0

    async def execute(self, data: object) -> object:
        """Выполняет имитацию запроса к ИИ.

        Args:
            data: Данные для обработки.

        Returns:
            Результат обработки.
        """
        self.attempts += 1

        # 1. Имитируем временный лимит на первую попытку
        if self.attempts == 1:
            print("Попытка 1: Имитируем Rate Limit...")
            raise RateLimitError("Rate limit exceeded", retry_delay=0.1)

        # 2. Имитируем блокировку контента для больших батчей
        if isinstance(data, list) and len(data) > 1:
            print(f"Батч из {len(data)} элементов заблокирован цензурой. Требуется деление!")
            raise ContentBlockedError("Sensitive content detected")

        print(f"Успешное выполнение для данных: {data}")
        return f"OK: {data}"

    def pack_single_prompt(self, prompt: str) -> object:
        """Упаковывает строку промпта."""
        return prompt


async def run_demo() -> None:
    """Запускает демонстрацию работы оркестратора с обработкой ошибок."""
    provider = FakeAIProvider()
    orchestrator = Orchestrator(provider=provider, strategy=RetryStrategy(max_retries=3, base_delay=0.1))

    print("--- Запуск оркестратора для батча из 3-х элементов ---")
    # Оркестратор сделает ретрай при первой ошибке,
    # а затем поделит батч, пока не найдет безопасные части.
    results = await orchestrator.execute(["Safe 1", "Safe 2", "Safe 3"])

    print("\n--- Финальные результаты ---")
    print(results)


if __name__ == "__main__":
    asyncio.run(run_demo())
