"""
Сквозной пример работы chllm:
1. Описание желаемой структуры через Pydantic.
2. Автоматическая генерация промпта с JSON-схемой через PromptBuilder.
3. Выполнение с циклом Self-Correction через Orchestrator и GenericCallableProvider.
4. Потоковый парсинг в реальном времени через RobustLLMParser.parse_stream.
"""

import asyncio

from pydantic import BaseModel, Field

from chllm import (
    GenericCallableProvider,
    Orchestrator,
    PromptBuilder,
    RobustLLMParser,
)


class DialogueLine(BaseModel):
    """Модель реплики персонажа."""

    speaker: str = Field(description="Имя говорящего персонажа")
    text: str = Field(description="Текст реплики")
    mood: str = Field(default="neutral", description="Настроение (happy, angry, calm, etc.)")


async def fake_llm_with_flaky_first_attempt(prompt: str) -> str:
    """Имитирует LLM, которая на первой попытке ошиблась, а на репромпте исправилась."""
    # Если в промпте есть пометка об ошибке от Orchestrator
    if "ПРЕДЫДУЩИЙ ОТВЕТ СОДЕРЖАЛ ОШИБКУ" in prompt:
        return """
        Вот исправленный JSON:
        ```json
        {
          "dialogues": [
            {"speaker": "Геральт", "text": "Зараза...", "mood": "calm"},
            {"speaker": "Лютик", "text": "Чеканной монетой!", "mood": "happy"}
          ]
        }
        ```
        """

    # Первая попытка: модель забыла закрыть скобку и вернула некорректный формат
    return """
    Конечно, вот диалог:
    {"dialogues": [{"speaker": "Геральт", "text": "Зараза
    """


async def run_structured_workflow_demo() -> None:
    """Демонстрация полного структурированного пайплайна."""
    print("=== 1. Генерация промпта с JSON-схемой через PromptBuilder ===")
    builder = PromptBuilder(template="Ты сценарист видеоигры. Сгенерируй диалог двух героев.")

    prompt = builder.build(
        input_data={"scene": "Таверна в Новиграде", "characters": ["Геральт", "Лютик"]},
        response_model=DialogueLine,
        context=["Герои только что победили чудовище."],
        context_label="ПРЕДЫСТОРИЯ",
    )
    print(prompt)
    print("\n" + "=" * 60 + "\n")

    print("=== 2. Выполнение запроса с Self-Correction через Orchestrator ===")
    provider = GenericCallableProvider(fake_llm_with_flaky_first_attempt)
    orchestrator = Orchestrator(provider=provider)

    # execute_structured автоматически повторит запрос, если первая попытка не прошла валидацию
    dialogues = await orchestrator.execute_structured(
        prompt=prompt,
        response_model=DialogueLine,
        container_keys=("dialogues", "items"),
    )

    print(f"Успешно получено реплик: {len(dialogues)}")
    for line in dialogues:
        print(f"[{line.mood.upper()}] {line.speaker}: {line.text}")
    print("\n" + "=" * 60 + "\n")

    print("=== 3. Потоковый парсинг в реальном времени (Streaming Parser) ===")
    parser = RobustLLMParser()

    # Имитируем поток токенов по сети
    async def token_stream():
        stream_chunks = [
            '{"dialogues": [',
            '{"speaker": "Йеннифэр",',
            ' "text": "Ты опоздал.",',
            ' "mood": "annoyed"}',
            ",",
            '{"speaker": "Геральт",',
            ' "text": "На плотве слетела подкова.",',
            ' "mood": "calm"}',
            "]}",
        ]
        for chunk in stream_chunks:
            await asyncio.sleep(0.05)
            yield chunk

    print("Начало потока...")
    async for dialogue in parser.parse_stream(token_stream(), validation_model=DialogueLine):
        print(f"-> Получен элемент из потока: {dialogue.speaker} ({dialogue.mood}): {dialogue.text}")


if __name__ == "__main__":
    asyncio.run(run_structured_workflow_demo())
