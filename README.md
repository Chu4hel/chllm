# chllm: Robust Structured Data Framework for LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`chllm` (произносится *chill-em*) — это профессиональный легковесный Python-фреймворк для построения отказоустойчивых конвейеров обработки данных через LLM. Библиотека специализируется на извлечении структурированных ответов (JSON / Pydantic) в условиях нестабильных API, обрывов контекста и жестких лимитов провайдеров.

---

## 🚀 Почему chllm?

Работа с LLM в реальных приложениях сопряжена с рядом проблем:
- **Обрезанные ответы**: модели часто не успевают закрыть скобки/кавычки JSON из-за лимита токенов.
- **Галлюцинации синтаксиса**: ИИ может непреднамеренно перевести или сломать переменные, плейсхолдеры и теги.
- **Сложные ошибки и Rate Limits**: ретраи для ошибок 429, 503 и фильтрации контента требуют принципиально разной обработки.
- **Агентные циклы**: необходимость надежно парсить вызовы инструментов (Tool Calls) и управлять шагами выполнения.

`chllm` берет всю эту рутину на себя.

---

## 📦 Установка

```bash
# Базовая установка (только Pydantic)
pip install chllm
# или через uv
uv add chllm

# С расширенным логированием через chutils
uv add "chllm[chutils]"

# С точным подсчетом токенов через tiktoken
uv add "chllm[tokens]"

# Полный набор
uv add "chllm[all]"
```

---

## 🛠 Ключевые модули

### 1. RobustLLMParser (`chllm.parser`)
Интеллектуальный парсер, способный извлекать и восстанавливать данные даже из поврежденных ответов:
- **Zero-Config Fallback**: автоматически инспектирует структуру полей Pydantic-модели и находит нужный массив объектов, даже если модель вернула неожиданное имя ключа.
- **Универсальные контейнеры**: поддержка явных параметров `container_key` и `container_keys` (например, `items`, `cards`, `dialogues`).
- **Стековое восстановление (Deep Recovery)**: автоматически достраивает незакрытые скобки, кавычки и массивы в оборванном JSON.
- **Извлечение из Markdown**: находит JSON-блоки внутри пояснительного текста или рассуждений модели.
- **Потоковый парсинг (`parse_stream`)**: асинхронный разбор чанков текста в реальном времени до завершения ответа модели.
- **Быстрый парсинг (`parse_items`)**: возвращает готовый типизированный `list[T]` без необходимости ручной распаковки словаря.
- **Tool Use Parsing**: метод `parse_tool_calls` находит структурированные вызовы инструментов.

### 2. Orchestrator & AgentOrchestrator (`chllm.orchestrator`)
Двигатель выполнения запросов с адаптивным поведением:
- **Цикл самоисправления (`execute_structured`)**: при ошибках валидации Pydantic автоматически формирует запрос на исправление и повторяет вызов до успеха.
- **Бинарное деление батчей (Batch Splitting)**: при возникновении ошибок размера или цензуры рекурсивно делит батч, изолируя сбойный элемент.
- **Умная стратегия повторов (RetryStrategy)**: экспоненциальная задержка с рандомизированным джиттером для защиты от перегрузки API.
- **Одиночные запросы (`execute_single`)**: универсальное извлечение текста из любых контейнеров (`dict` или объектов) и полей (`content`, `text`, `message`, `output` и др.).
- **Агентный цикл (`AgentOrchestrator`)**: метод `execute_tools` берет на себя выполнение вызовов инструментов.

### 3. Провайдеры-адаптеры (`chllm.providers`)
Готовые провайдеры для быстрого старта с протоколом `LLMProvider`:
- `GenericCallableProvider`: оборачивает любую функцию или корутину `async def (payload) -> response`.
- `OpenAICompatibleProvider`: адаптер для любых OpenAI-совместимых клиентов (`AsyncOpenAI`, LiteLLM, vLLM, Ollama, DeepSeek).

### 4. Prompt & Context Builders (`chllm.builder`, `chllm.context`)
- `PromptBuilder`: динамическая сборка промптов, контекста и данных, а также автоматическая генерация инструкций со строгой JSON-схемой из Pydantic-моделей (`response_model`).
- `ContextBuilder`: управление цепочкой контекста диалогов (Chain Context / Full Context).

### 5. ContentMasker (`chllm.masking`)
Защита системного синтаксиса и чувствительных участков текста:
- Маскирует переменные (например, `[MCname]`, `%(user)s`, `{b}...{/b}`) в плейсхолдеры вида `[[[VAR_0]]]`.
- Модель видит структуру предложения, но физически не может повредить или перевести системные теги.
- Корректная сортировка паттернов по длине для предотвращения коллизий.

### 6. Metrics & Token Estimation (`chllm.metrics`)
Контроль расхода токенов:
- `TokenCounter`: поддержка эвристического расчета для русского и английского языков, а также токенизатора `tiktoken`.
- `estimate_completion_tokens`: прогнозирование объема ответа с учетом коэффициента языкового расширения и оверхеда схемы.

---

## 📖 Быстрый старт

### Восстановление поврежденного JSON

```python
from pydantic import BaseModel
from chllm import RobustLLMParser


class UserItem(BaseModel):
    id: int
    name: str


parser = RobustLLMParser()

# Модель оборвала ответ на середине:
broken_response = """
Вот результаты:
```json
[
  {"id": 1, "name": "Алиса"},
  {"id": 2, "name": "Борис"
"""

data = parser.parse(broken_response, validation_model=UserItem)
# data["batch"] -> [UserItem(id=1, name="Алиса")]
```

### Защита переменных при переводе / рерайте

```python
from chllm import ContentMasker

masker = ContentMasker(patterns=[r"\[.+?\]", r"\{.+?\}"])
text = "Привет, [player_name]! Нажми {b}Старт{/b}."

masked = masker.mask(text)
# masked.masked_text -> "Привет, [[[VAR_0]]]! Нажми [[[VAR_1]]]Старт[[[VAR_2]]]."

# Отправляем masked.masked_text в LLM и получаем "Hello, [[[VAR_0]]]! Press [[[VAR_1]]]Start[[[VAR_2]]]."

demasked = masker.demask(translated_text, masked.mapping)
# demasked -> "Hello, [player_name]! Press {b}Start{/b}."
```

### Структурированный запрос с самоисправлением (Self-Correction)

```python
from pydantic import BaseModel
from chllm import GenericCallableProvider, Orchestrator, PromptBuilder


class Card(BaseModel):
    title: str
    points: int


# Генерируем промпт со строгой JSON-схемой модели
builder = PromptBuilder()
prompt = builder.build(
    input_data={"theme": "Фэнтези"},
    response_model=Card,
)

# Оборачиваем функцию вызова API
provider = GenericCallableProvider(my_async_llm_function)
orchestrator = Orchestrator(provider)

# При повреждении JSON или ошибке валидации оркестратор автоматически сделает репромпт
cards = await orchestrator.execute_structured(prompt, response_model=Card)
# cards -> [Card(title="Рыцарь", points=10), ...]
```

### Потоковый парсинг в реальном времени (Streaming Parser)

```python
from chllm import RobustLLMParser

parser = RobustLLMParser()

# Получаем готовые объекты прямо во время генерации токенов
async for card in parser.parse_stream(my_token_stream, validation_model=Card):
    print(f"Новая карточка: {card.title} ({card.points} очков)")
```

### Оркестратор запросов с ретраями

```python
from chllm import Orchestrator, RetryStrategy, RateLimitError


class MyLLMProvider:
    async def execute(self, payload: str) -> str:
        # Ваш сетевой вызов к API модели
        return await api_client.generate(payload)


orchestrator = Orchestrator(
    provider=MyLLMProvider(),
    strategy=RetryStrategy(max_retries=3, base_delay=1.5),
)

response = await orchestrator.execute_single("Объясни квантовую запутанность кратко.")
```

---

## 🏗 Архитектура

Библиотека строго следует принципу **Dependency Inversion**:
- Модули не привязаны к конкретным внешним SDK (Google GenAI, OpenAI, Anthropic) — взаимодействие построено через протоколы.
- Логирование автономно: при наличии `chutils` используется его структурированный логгер, иначе — стандартный `logging`.

---

## 📄 Лицензия

Распространяется под лицензией [MIT](LICENSE).
