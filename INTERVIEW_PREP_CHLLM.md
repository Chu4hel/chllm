# Опенсорс библиотека на python

<https://github.com/Chu4hel/chllm>

# chllm: Досье проекта для технического интервью (Python / AI & LLM Backend Developer)

> **Назначение документа:** Комплексная инженерная выжимка и шпаргалка по архитектуре, кодовой базе, алгоритмам отказоустойчивости и техническим компромиссам фреймворка `chllm` (v0.3.0).\\
> **Целевая аудитория:** Кандидат на позиции **Middle+ / Senior Python Developer**, **AI / LLM Backend Engineer**, **Tech Lead**.\\
> **Статус:** Локальный рабочий документ для портфолио и подготовки к техническим интервью.

---

## 1. Карточка проекта (Elevator Pitch за 60 секунд)

- **Продукт:** `chllm` (*произносится как «chill-em»*) — высокопроизводительный, модульный и отказоустойчивый Python-фреймворк для извлечения гарантированно валидных структурированных данных (JSON / Pydantic) из LLM, мультиагентной оркестрации, управления пулами провайдеров и обработки сверхдлинных контекстов.
- **Специализация:** Решение прикладных проблем Production LLM-пайплайнов:
  - Обрыв генерации по токенам и автоматическое стековое восстановление битого JSON.
  - Умная ротация пулов API-ключей (Rate Limits 429, квоты, экспоненциальный Full Jitter backoff).
  - Изоляция заблокированного цензурой контента в батчах через бинарный рекурсивный поиск.
  - Двустороннее маскирование переменных и системного синтаксиса (`[[[VAR_0]]]`).
  - Потоковая real-time валидация Pydantic по мере генерации токенов.
  - Мультиагентные сессии (ConversationPool, форкинг контекста, изолированные шины сообщений).
  - Интеллектуальный чанкинг и Map-Reduce суммаризация гигантских документов.
  - Архитектура динамических навыков (Skills) с валидацией схем и CLI-инструментарием.
- **Автор:** Chu4hel ([sergeiivanov636@gmail.com](mailto:sergeiivanov636@gmail.com))
- **Стек технологий:**
  - **Ядро:** Python 3.10+, Pydantic v2 (Rust core, строгие схемы, `TypeAdapter`, `model_validate`, `model_json_schema`), PyYAML 6.0+, `asyncio` (Semaphore, gather, Locks, Tasks).
  - **Архитектурные паттерны:** Clean Architecture, Dependency Inversion (`typing.Protocol`), Zero Core Bloat, Stack Machine JSON Repair, Self-Correction via Reflection, Binary Batch Splitting, Multi-Provider Key Rotation, Map-Reduce Chaining.
  - **Инструменты сборки и качества:** uv (пакетный менеджер), hatchling (PEP 517/621 build backend), pytest + pytest-asyncio + pytest-mock (100% изоляция без реальных платных API), ruff (linter/formatter), mypy (строгая типизация, PEP 561 py.typed).
  - **Опциональные модули:**
    - `chllm[gemini]`: `google-genai>=0.1.0` (официальный SDK Google GenAI, поддержка Thinking Mode).
    - `chllm[router]`: `httpx>=0.27.0` (OpenRouter, корпоративные AI шлюзы и кастомные прокси).
    - `chllm[tokens]`: `tiktoken>=0.7.0` (высокоточный подсчет токенов BPE).
    - `chllm[chutils]`: `chutils>=3.5.0` (структурированное логирование с маскированием PII).

> **Фраза для самопрезентации:**\\
> *«`chllm` — это open-source фреймворк, который я спроектировал для решения главных болей промышленной AI-разработки: нестабильности ответов LLM, исчерпания лимитов API и сложности интеграции агентов. В отличие от перегруженных комбайнов вроде LangChain, `chllm` построен по принципам чистой архитектуры и решает задачи со строгой инженерной гарантией: восстанавливает обрезанный JSON стековым автоматом, автоматически ротирует пулы ключей при ошибках 429, изолирует заблокированный цензурой контент методом бинарного деления, реализует Map-Reduce над документами любого объема и предоставляет типобезопасную мультиагентную среду с системой динамических навыков.»*

---

## 2. Архитектура и сквозной поток данных

Фреймворк спроектирован по строгим принципам **Dependency Inversion (DIP)** и **Loose Coupling**. Доменная логика, алгоритмы нормализации и оркестрации абстрагированы от сетевых протоколов и SDK конкретных провайдеров.

### 2.1. Диаграмма подсистем фреймворка

```mermaid
flowchart TD
    subgraph Ingestion["1. Подготовка и декомпозиция данных"]
        RawInput["Входной текст / Документ / Батч"]
        Chunker["TextChunker / TokenChunker\n(нарезка по токенам + иерархия разделителей)"]
        Summarizer["HierarchicalProcessor / ChainedSummarizer\n(Map-Reduce рекурсивное сжатие)"]
        Masker["ContentMasker\n(пресеты MaskingPresets, композиция через |)"]
        Builder["PromptBuilder\n(Few-Shot примеры + автогенерация JSON-схемы)"]
    end

    subgraph AgenticCore["2. Агентный слой и навыки (chllm.agentic & chllm.skills)"]
        ConvPool["ConversationPool & AgentSession\n(управление сессиями, ForkMode)"]
        MsgBus["SessionMessageBus (шина сообщений)"]
        SkillReg["SkillRegistry & SkillValidator\n(декларативные YAML навыки)"]
        CliRunner["CliTool (безопасное исполнение подпроцессов)"]
    end

    subgraph Orchestration["3. Оркестрация и отказоустойчивость"]
        Orchestrator["Orchestrator / AgentOrchestrator"]
        RetryMngr["RetryStrategy (Full Jitter Exponential Backoff)"]
        BatchSplitter["Binary Batch Splitting (изоляция ContentBlockedError)"]
        Metrics["UsageMetrics & TokenCounter"]
    end

    subgraph ProviderLayer["4. Сетевой слой и провайдеры (LLMProvider)"]
        KeyPool["KeyPoolProvider & MultiKeyRotator\n(Round-Robin / Least-Used, Cooldown)"]
        OpenAIAdapter["OpenAICompatibleProvider (vLLM, Ollama, DeepSeek)"]
        GoogleAdapter["GoogleGenAIProvider (Gemini Thinking Mode)"]
        RouterAdapter["OpenAIRouterProvider (OpenRouter, httpx proxy)"]
        GenericAdapter["GenericCallableProvider (корутины)"]
    end

    subgraph ParsingLayer["5. Интеллектуальный парсинг & Валидация"]
        RawStream["Сырой ответ / Network Stream"]
        ConvParse["parse_conversational_response (очистка CoT и мыслей)"]
        SMRepair["_json_repair (стековый автомат LIFO)"]
        ZeroConfig["Zero-Config Matching (скоринг полей)"]
        PydanticVal["Pydantic v2 (model_validate)"]
        SelfCorrect{"Успех валидации?"}
    end

    RawInput --> Chunker --> Summarizer --> Masker --> Builder --> Orchestrator
    ConvPool <--> MsgBus
    ConvPool --> Orchestrator
    SkillReg --> ConvPool
    CliRunner --> ConvPool

    Orchestrator --> RetryMngr --> KeyPool
    KeyPool --> OpenAIAdapter & GoogleAdapter & RouterAdapter & GenericAdapter
    OpenAIAdapter & GoogleAdapter & RouterAdapter & GenericAdapter --> RawStream

    RawStream --> ConvParse --> SMRepair --> ZeroConfig --> PydanticVal --> SelfCorrect
    SelfCorrect -- "Ошибка (попытка < max)" -->|Self-Correction Prompt| Orchestrator
    SelfCorrect -- "Успех" --> Output["Готовые Pydantic-модели + Demasking"]
    Orchestrator -.-> Metrics
    OpenAIAdapter -.->|ContentBlockedError| BatchSplitter -.->|50/50 деление| Orchestrator
```

---

## 3. Глубокий разбор ключевых инженерных компонентов

### 3.1. Пул API-ключей и многоуровневая ротация (`KeyPoolProvider`, `MultiKeyRotator`)
- **Проблема:** Высокая производственная нагрузка быстро упирается в Rate Limits (RPM/TPM) и квоты провайдеров (Google Gemini, Groq, OpenRouter).
- **Решение:** `KeyPoolProvider` управляет пулом `KeyItem` с отслеживанием жизненного цикла (`KeyStatus.ACTIVE`, `COOLDOWN`, `EXHAUSTED`, `BANNED`).
- **Стратегии ротации:** `ROUND_ROBIN`, `LEAST_USED`, `PRIORITY`.
- **Отказоустойчивость:** При получении `RateLimitError` (429) ключ автоматически переводится в `COOLDOWN` на расчетное время `retry_after`, а запрос прозрачно перенаправляется на следующий доступный ключ без сбоя пайплайна.

### 3.2. Мультиагентные сессии и шина сообщений (`chllm.agentic`)
- **ConversationPool:** Пул агентных контекстов с поддержкой режимов ветвления (`ForkMode`):
  - `ISOLATED`: Полная изоляция контекста.
  - `INHERIT_PROMPT`: Наследование системного промпта и ролевой модели.
  - `FORK_HISTORY`: Глубокое клонирование всей истории диалога.
- **SessionMessageBus:** Асинхронная шина сообщений между сессиями (`AgentMessage`) с фильтрацией адресатов и историей переписки.
- **ISessionStorage:** Абстракция персистентного хранения сессий (`SessionSnapshot`) с возможностью сериализации в Redis/БД или памяти (`InMemorySessionStorage`).

### 3.3. Архитектура динамических навыков (`chllm.skills`)
- **Декларативные манифесты:** Описание навыков через стандартизированный формат YAML/Pydantic (`SkillDefinition`, `SkillMetadata`).
- **Источники навыков (ISkillSource):** `FileSystemSkillSource` для динамической загрузки из директорий и `InMemorySkillSource` для рантайм-регистрации.
- **SkillRegistry & SkillValidator:** Строгая проверка схемы аргументов инструментов и зависимостей перед инъекцией в контекст агента.

### 3.4. Иерархическая обработка и Map-Reduce суммаризация (`chunker.py`, `summarizer.py`)
- **TextChunker / TokenChunker:** Смысловая нарезка текстов с контролем токенов (`ChunkingConfig`), настраиваемым `overlap` и иерархией разделителей (`\n\n`, `\n`, предложения, пробелы).
- **HierarchicalProcessor & ChainedSummarizer:** Параллельный Map-шаг через `AsyncBatchProcessor` с последующей многоуровневой Reduce-сверткой промежуточных чанков. Позволяет суммаризировать книги и базы знаний на 1 000 000+ токенов в рамках бюджетных моделей.

### 3.5. Безопасное исполнение CLI-инструментов (`chllm.tools`)
- **CliTool & CliToolConfig:** Вызов локальных утилит с контролем таймаутов (`CliTimeoutError`), режимами передачи аргументов (`CliInputMode.ARGV`, `STDIN`, `ENV`) и изолированным захватом stdout/stderr.

### 3.6. Стековый автомат восстановления поврежденного JSON (`_json_repair.py`, `parser.py`)
- Двухфазный алгоритм: посимвольный поиск границ сбалансированных JSON-блоков (`find_json_blocks`) с игнорированием кавычек и экранирования, плюс LIFO-стековое закрытие скобок и кавычек при обрыве токенов (`attempt_json_recovery`).
- Рекурсивный поиск уцелевших объектов внутри поврежденных массивов (`blocks_queue`).
- Очистка CoT и мыслей рассуждающих моделей (`parse_conversational_response`).

### 3.7. Эвристический Zero-Config Fallback и Model Matching (`parser.py`)
- Автоматический расчет коэффициента пересечения ключей модели Pydantic (с учетом `validation_alias`) и найденных списков словарей в ответе LLM. Не требует жесткой настройки имени контейнера (`Zero-Config`).

### 3.8. Бинарное деление батчей при срабатывании цензуры (`orchestrator.py`)
- При перехвате `ContentBlockedError` батч делится 50/50 с логарифмической сложностью $O(\log N)$. Проблемный элемент изолируется, а остальные 99% батча успешно обрабатываются.

### 3.9. Защита системного синтаксиса и плейсхолдеров (`masking.py`)
- `ContentMasker` с поддержкой `MaskingPresets` (email, url, code, custom) и удобной композицией через бинарный оператор: `preset = MaskingPresets.CODE | MaskingPresets.URL`.
- Детерминированная замена на `[[[VAR_0]]]` в один проход ($O(M)$) через объединенный regex.

---

## 4. Сводная таблица архитектурных модулей

| Модуль | Ключевые классы / компоненты | Назначение и инженерные решения |
|---|---|---|
| **`chllm.agentic`** | `ConversationPool`, `AgentSession`, `SessionMessageBus`, `ForkMode` | Мультиагентная оркестрация, ветвление диалоговых сессий, шина обмена сообщениями. |
| **`chllm.skills`** | `SkillRegistry`, `SkillValidator`, `FileSystemSkillSource`, `SkillDefinition` | Декларативные навыки, валидация сигнатур функций и изоляция ошибок. |
| **`chllm.providers`** | `KeyPoolProvider`, `GoogleGenAIProvider`, `OpenAIRouterProvider`, `OpenAICompatibleProvider` | Пулы ключей с ротацией (Cooldown, Least-Used), поддержка Gemini Thinking Mode и шлюзов OpenRouter. |
| **`chllm.chunker`** | `TextChunker`, `TokenChunker`, `ChunkingConfig` | Токено-ориентированная нарезка текстов с перекрытием и иерархией разделителей. |
| **`chllm.summarizer`**| `HierarchicalProcessor`, `ChainedSummarizer`, `MapReduceConfig` | Рекурсивная и параллельная Map-Reduce суммаризация сверхдлинных контекстов. |
| **`chllm.tools`** | `CliTool`, `CliToolConfig`, `CliInputMode` | Безопасное выполнение CLI подпроцессов в песочнице с таймаутами. |
| **`chllm.orchestrator`** | `Orchestrator`, `AgentOrchestrator`, `RetryStrategy` | Оркестрация вызовов, Exponential Backoff c джиттером, бинарное деление батчей. |
| **`chllm.parser`** | `RobustLLMParser`, `_json_repair` | LIFO-стековое восстановление оборванного JSON, Zero-Config матчинг, стриминг `parse_stream`. |
| **`chllm.masking`** | `ContentMasker`, `MaskingPresets` | Защита плейсхолдеров и переменных, оператор композиции `|`. |
| **`chllm.metrics`** | `TokenCounter`, `UsageMetrics` | Мониторинг задержек, расчет стоимости и токенов prompt/completion. |

---

## 5. Вопросы и ответы (Q&A) для технического интервью

### В1: Как в `chllm` решена проблема Rate Limits (429) и исчерпания суточных квот?
**Ответ кандидата:**
*«На сетевом уровне реализован `KeyPoolProvider`. Он агрегирует пул ключей с поддержкой стратегий `ROUND_ROBIN`, `LEAST_USED` и `PRIORITY`. Если провайдер возвращает 429 или исчерпание квоты (`QuotaExceededError`), ключ помечается статусом `COOLDOWN` на заданный интервал или время из заголовка `retry-after`, а запрос прозрачно переключается на следующий живой ключ. На уровне единичного запроса работает `RetryStrategy` с алгоритмом Full Jitter Exponential Backoff, исключающим Thundering Herd Problem.»*

### В2: Как обрабатывать документы на сотни тысяч токенов, если контекстное окно ограничено?
**Ответ кандидата:**
*«Для этого спроектированы модули `chllm.chunker` и `chllm.summarizer`. `TextChunker` выполняет семантическую нарезку с контролем токенов по BPE (`tiktoken`) с сохранением абзацев и настраиваемым overlap. Затем `HierarchicalProcessor` выполняет параллельный Map-этап через асинхронный семафор `AsyncBatchProcessor`, после чего каскадно объединяет и сжимает промежуточные результаты на Reduce-слоях, пока итоговый текст не уложится в целевой лимит.»*

### В3: Чем агентная система в `chllm` отличается от монолитных агентов?
**Ответ кандидата:**
*«В `chllm.agentic` агентная среда декомпозирована на легковесные изолированные сущности. Сессиями управляет `ConversationPool`, позволяющий делать легковесный форк контекста (`ForkMode.FORK_HISTORY` или `INHERIT_PROMPT`). Взаимодействие агентов построено через шину `SessionMessageBus`, а состояние может сериализоваться в снимки `SessionSnapshot`. Внешние возможности подключаются через валидируемый реестр навыков `SkillRegistry` и защищенный запуск CLI-утилит `CliTool` с контролем таймаутов.»*

### В4: Как устроен стековый парсер восстановления JSON?
**Ответ кандидата:**
*«Алгоритм работает в 2 фазы: сначала конечный автомат `find_json_blocks` посимвольно выделяет границы сбалансированных структур, отслеживая состояние строк и экранирование кавычек, что отсекает CoT-болтовню модели. Если JSON оборван по лимиту токенов, `attempt_json_recovery` обрезает висячие запятые/двоеточия, закрывает незавершенную строку и выталкивает из LIFO-стека недостающие фигурные и квадратные скобки. Для Pydantic работает эвристический Zero-Config скоринг полей.»*

---

## 6. Чек-лист сильных тем (Talking Points)

- [x] **Zero-Bloat Core vs Heavy Frameworks:** Минималистичное ядро на Pydantic v2 и PyYAML с ленивыми расширениями.
- [x] **Key Pool & High Availability:** Управление пулами ключей, автоматический cooldown и ротация при 429.
- [x] **Hierarchical Map-Reduce:** Обработка гигантских объемов текста с сохранением контекста.
- [x] **Resilience & Chaos-Proof Parsing:** Стековое восстановление оборванных ответов и изоляция цензуры бинарным делением.
- [x] **Dynamic Skills & Agent Bus:** Модульная архитектура навыков с валидацией контрактов и асинхронная шина агентов.\n