"""Провайдер для работы с Google Gemini через SDK google-genai."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, NoReturn

from pydantic import BaseModel, Field

from ..exceptions import (
    AuthenticationError,
    CHLLMError,
    ContentBlockedError,
    LocationUnsupportedError,
    ProxyConnectionError,
    QuotaExceededError,
    RateLimitError,
    ServiceUnavailableError,
)

try:
    from chutils.logger import setup_logger
except ImportError:

    def setup_logger(name: str = "chllm.providers.google_genai") -> Any:
        """Создает резервный логгер, если chutils не установлен."""
        return logging.getLogger(name)


def model_supports_thinking(model_name: str) -> bool:
    """Проверяет, поддерживает ли модель управление параметрами thinking.

    Args:
        model_name: Имя модели Gemini.

    Returns:
        True, если модель поддерживает thinking, иначе False.
    """
    if not model_name:
        return False
    name = model_name.lower()
    return "thinking" in name or "2.5" in name or "3." in name or "gemini-3" in name


class GeminiThinkingConfig(BaseModel):
    """Конфигурация параметров размышлений (thinking) модели Gemini."""

    thinking_level: str | None = Field(
        default=None,
        description="Уровень размышлений (например, 'LOW', 'MEDIUM', 'HIGH', 'MINIMAL').",
    )
    thinking_budget: int | None = Field(
        default=None,
        description="Бюджет токенов на размышления (0 для принудительного отключения).",
    )


class GoogleGenAIProvider:
    """Провайдер для работы с Google Gemini через SDK google-genai.

    Обеспечивает интеллектуальный разбор сетевых ошибок и квот (429 RPM vs Quota, гео-блокировки, прокси),
    поддержку динамического плавающего таймаута и автоматический fallback параметров thinking.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        client: Any | None = None,
        system_instruction: str | None = None,
        thinking_config: GeminiThinkingConfig | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 65536,
        proxy_url: str | None = None,
        base_timeout: float = 120.0,
        min_timeout: float = 180.0,
        max_timeout: float = 1200.0,
        tokens_per_second: float = 18.0,
        logger: Any | None = None,
    ) -> None:
        """Инициализирует GoogleGenAIProvider.

        Args:
            model: Имя модели Gemini (например, 'gemini-2.5-flash').
            api_key: API ключ Google Gemini.
            client: Экземпляр genai.Client (если уже создан).
            system_instruction: Системная инструкция по умолчанию.
            thinking_config: Параметры рассуждений (thinking).
            temperature: Температура генерации.
            max_output_tokens: Максимальное количество токенов в ответе.
            proxy_url: URL прокси-сервера.
            base_timeout: Базовый резерв времени на таймаут (сек).
            min_timeout: Минимальный порог таймаута (сек).
            max_timeout: Максимальный потолок таймаута (сек).
            tokens_per_second: Пессимистичная скорость генерации (токенов/сек).
            logger: Пользовательский логгер.
        """
        self.model = model
        self.api_key = api_key
        self.system_instruction = system_instruction
        self.thinking_config = thinking_config
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.proxy_url = proxy_url
        self.base_timeout = base_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.tokens_per_second = tokens_per_second
        self.logger = logger or setup_logger("chllm.providers.google_genai")

        if client is not None:
            self._client = client
        else:
            self._client = self._init_client()

    def _init_client(self) -> Any:
        """Инициализирует клиент google-genai с настройками прокси и HTTP."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as err:
            raise ImportError(
                "Для использования GoogleGenAIProvider необходимо установить пакет google-genai: "
                "pip install 'chllm[gemini]' или pip install google-genai"
            ) from err

        http_options = None
        if self.proxy_url:
            httpx_config: dict[str, Any] = {
                "http2": False,
                "headers": {"Connection": "close"},
                "proxy": self.proxy_url,
            }
            http_options = types.HttpOptions(
                api_version="v1beta",
                client_args=httpx_config,
                async_client_args=httpx_config,
            )

        return genai.Client(api_key=self.api_key, http_options=http_options)

    def calculate_dynamic_timeout(
        self,
        expected_output_tokens: int,
        base_timeout: float | None = None,
        min_timeout: float | None = None,
        max_timeout: float | None = None,
        tokens_per_second: float | None = None,
    ) -> float:
        """Рассчитывает плавающий таймаут ожидания ответа ИИ в зависимости от объема вывода.

        Args:
            expected_output_tokens: Оценочное количество выходных токенов.
            base_timeout: Базовый резерв времени (сек).
            min_timeout: Минимальный порог таймаута (сек).
            max_timeout: Максимальный потолок таймаута (сек).
            tokens_per_second: Пессимистичная скорость генерации (токенов/сек).

        Returns:
            Рассчитанный таймаут в секундах.
        """
        base = self.base_timeout if base_timeout is None else base_timeout
        min_t = self.min_timeout if min_timeout is None else min_timeout
        max_t = self.max_timeout if max_timeout is None else max_timeout
        tps = self.tokens_per_second if tokens_per_second is None else tokens_per_second

        dynamic_val = base + (expected_output_tokens / tps)
        return max(min_t, min(dynamic_val, max_t))

    def _build_config(
        self,
        system_instruction: str | None = None,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
    ) -> Any:
        """Создает объект конфигурации генерации для Google GenAI SDK.

        Args:
            system_instruction: Системная инструкция.
            thinking_level: Уровень размышлений (LOW, MEDIUM, HIGH, MINIMAL).
            thinking_budget: Бюджет токенов на размышления.

        Returns:
            Объект GenerateContentConfig или аналогичная структура.
        """
        sys_prompt = system_instruction if system_instruction is not None else self.system_instruction

        try:
            from google.genai import types

            thinking_config: types.ThinkingConfig | None = None
            if thinking_level is not None or thinking_budget is not None:
                level: types.ThinkingLevel | None = None
                if thinking_level is not None:
                    if isinstance(thinking_level, types.ThinkingLevel):
                        level = thinking_level
                    elif isinstance(thinking_level, str):
                        try:
                            level = types.ThinkingLevel[thinking_level.upper()]
                        except (KeyError, AttributeError):
                            level = None
                thinking_config = types.ThinkingConfig(thinking_level=level, thinking_budget=thinking_budget)

            return types.GenerateContentConfig(
                system_instruction=sys_prompt if sys_prompt else None,
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
                thinking_config=thinking_config,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
            )
        except ImportError:
            # Fallback для тестов и сред без google-genai
            cfg: dict[str, Any] = {
                "system_instruction": sys_prompt,
                "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature,
            }
            if thinking_level is not None or thinking_budget is not None:
                cfg["thinking_config"] = {
                    "thinking_level": thinking_level,
                    "thinking_budget": thinking_budget,
                }
            return cfg

    def handle_api_error(self, e: BaseException) -> NoReturn:
        """Универсальный метод интеллектуального разбора ошибок от Gemini SDK.

        Args:
            e: Исходное исключение.

        Raises:
            ProxyConnectionError: При сетевых сбоях, обрывах прокси или таймаутах.
            QuotaExceededError: При исчерпании суточной квоты.
            RateLimitError: При превышении минутного лимита запросов (RPM/TPM).
            ServiceUnavailableError: При перегрузке или недоступности серверов Gemini (503).
            LocationUnsupportedError: При блокировке по геолокации.
            AuthenticationError: При недействительном API-ключе.
            ContentBlockedError: При срабатывании фильтров безопасности.
            CHLLMError: При прочих ошибках API.
        """
        # 1. Если это таймаут или прерывание запроса
        if isinstance(e, (asyncio.TimeoutError, TimeoutError, asyncio.CancelledError)):
            raise ProxyConnectionError(f"Таймаут или прерывание запроса ИИ: {type(e).__name__}") from e

        # 2. Если исключение уже типизировано как ошибка библиотеки, пробрасываем
        if isinstance(
            e,
            (
                RateLimitError,
                QuotaExceededError,
                ServiceUnavailableError,
                LocationUnsupportedError,
                ProxyConnectionError,
                ContentBlockedError,
                AuthenticationError,
            ),
        ):
            raise e

        # 3. Рекурсивный поиск корневой причины ошибки
        root_error: BaseException = e
        visited: set[int] = {id(root_error)}
        while True:
            candidate = root_error.__cause__ or root_error.__context__
            if candidate is None or id(candidate) in visited:
                break
            visited.add(id(candidate))
            root_error = candidate

        error_type_name = type(e).__name__.lower()
        root_type_name = type(root_error).__name__.lower()
        error_str = str(e).lower()
        root_str = str(root_error).lower()
        combined_text = f"{error_str} {root_str}"

        # 4. Проверка сетевых сбоев и ошибок прокси
        net_signatures = (
            "remoteprotocolerror",
            "connecterror",
            "connecttimeout",
            "readtimeout",
            "networkerror",
            "proxyerror",
            "readerror",
            "timeoutexception",
            "connectionreseterror",
            "connectionrefusederror",
        )
        is_net_error = (
            any(sig in error_type_name for sig in net_signatures)
            or any(sig in root_type_name for sig in net_signatures)
            or "server disconnected without sending a response" in combined_text
            or "connection reset" in combined_text
            or "connection refused" in combined_text
        )
        if is_net_error:
            msg = f"Прокси/сетевая ошибка: {type(root_error).__name__} - {root_error!s}"
            raise ProxyConnectionError(msg) from e

        # 5. Ошибки аутентификации
        auth_signatures = ("api_key_invalid", "unauthenticated", "permission_denied", "invalid api key")
        if any(sig in combined_text for sig in auth_signatures):
            raise AuthenticationError(f"Ошибка аутентификации Gemini API: {e}") from e

        # 6. Блокировка по геолокации
        if "user location is not supported" in combined_text:
            raise LocationUnsupportedError(f"Доступ заблокирован по геолокации (User location unsupported): {e}") from e

        # 7. Недоступность сервиса (503 / Unavailable / Overloaded)
        retry_delay_match = re.search(
            r'["\']?retryDelay["\']?:\s*["\']?(\d+(?:\.\d+)?)s?["\']?',
            combined_text,
            re.IGNORECASE,
        )
        has_retry_delay = retry_delay_match is not None

        is_503 = (
            "503" in combined_text
            or "unavailable" in combined_text
            or "overloaded" in combined_text
            or getattr(e, "code", None) == 503
        )
        if is_503:
            retry_delay = 30.0
            if retry_delay_match:
                try:
                    retry_delay = float(retry_delay_match.group(1))
                except ValueError:
                    retry_delay = 30.0

            raise ServiceUnavailableError(
                f"Сервис Gemini временно перегружен или недоступен (503). Повтор через {retry_delay}с.",
                retry_delay=retry_delay,
            ) from e

        # 8. Суточная квота vs минутный лимит (429)
        # В Google GenAI статус 429 часто содержит "RESOURCE_EXHAUSTED".
        # Наличие retryDelay или упоминаний RPM/TPM/rate limit указывает на временный минутный лимит.
        rate_limit_keywords = ("rate limit", "rpm", "tpm", "per minute", "too many requests")
        is_explicit_rate_limit = any(kw in combined_text for kw in rate_limit_keywords)

        daily_quota_signatures = [
            "per day",
            "daily quota",
            "day quota",
            "quota exceeded",
            "free tier quota exhausted",
        ]
        is_daily_quota = (
            any(sig in combined_text for sig in daily_quota_signatures)
            and not has_retry_delay
            and not is_explicit_rate_limit
        )

        if is_daily_quota:
            raise QuotaExceededError(f"Исчерпана суточная квота Gemini API: {e}") from e

        is_429 = (
            "429" in combined_text
            or getattr(e, "code", None) == 429
            or has_retry_delay
            or is_explicit_rate_limit
            or "resource_exhausted" in combined_text
            or "exhausted" in combined_text
        )

        if is_429:
            retry_delay = 60.0
            if retry_delay_match:
                try:
                    retry_delay = float(retry_delay_match.group(1))
                except ValueError:
                    retry_delay = 60.0

            raise RateLimitError(
                f"Превышен минутный лимит запросов Gemini API (RPM/TPM). Пауза {retry_delay}с.",
                retry_delay=retry_delay,
            ) from e

        # 9. Блокировка контента фильтрами безопасности
        if "safety" in combined_text or "prompt_feedback" in combined_text or "blocked" in combined_text:
            raise ContentBlockedError(f"Контент заблокирован политиками безопасности Gemini: {e}") from e

        # 10. Общий fallback
        raise CHLLMError(f"Ошибка вызова Google GenAI: {type(e).__name__} - {e}") from e

    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
        expected_output_tokens: int | None = None,
    ) -> str:
        """Выполняет запрос к Gemini API с динамическим таймаутом и fallback для thinking.

        Args:
            prompt: Текст пользовательского запроса.
            system_instruction: Опциональная системная инструкция.
            thinking_level: Уровень размышлений (LOW, MEDIUM, HIGH, MINIMAL).
            thinking_budget: Бюджет токенов на размышления (0 для отключения).
            expected_output_tokens: Ожидаемый объем ответа для расчета таймаута.

        Returns:
            Сгенерированный текст ответа модели.
        """
        # 1. Определение параметров thinking
        supports_thinking = model_supports_thinking(self.model)
        eff_level = thinking_level
        eff_budget = thinking_budget

        if supports_thinking:
            if eff_level is None and eff_budget is None and self.thinking_config is not None:
                eff_level = self.thinking_config.thinking_level
                eff_budget = self.thinking_config.thinking_budget
        else:
            eff_level = None
            eff_budget = None

        config = self._build_config(
            system_instruction=system_instruction,
            thinking_level=eff_level,
            thinking_budget=eff_budget,
        )

        # 2. Расчет динамического таймаута
        est_tokens = expected_output_tokens if expected_output_tokens is not None else max(1500, int(len(prompt) / 4.0))
        dynamic_timeout = self.calculate_dynamic_timeout(est_tokens)

        # 3. Вызов API с автоматическим повтором при сбое thinking
        try:
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config,
                    ),
                    timeout=dynamic_timeout,
                )
            except Exception as call_err:
                if (eff_level is not None or eff_budget is not None) and "thinking" in str(call_err).lower():
                    self.logger.warning(
                        "Модель %s отклонила thinking_config: %s. Повтор без параметров thinking.",
                        self.model,
                        call_err,
                    )
                    fallback_config = self._build_config(
                        system_instruction=system_instruction,
                        thinking_level=None,
                        thinking_budget=None,
                    )
                    response = await asyncio.wait_for(
                        self._client.aio.models.generate_content(
                            model=self.model,
                            contents=prompt,
                            config=fallback_config,
                        ),
                        timeout=dynamic_timeout,
                    )
                else:
                    raise call_err

            # 4. Проверка блокировки или пустого ответа
            if not response or not getattr(response, "text", None):
                feedback = getattr(response, "prompt_feedback", None)
                if feedback:
                    block_msg = f"Ответ от Gemini пустой. Фидбек безопасности: {feedback}"
                else:
                    block_msg = "Ответ от Gemini пустой."
                raise ContentBlockedError(block_msg)

            return str(response.text)

        except Exception as e:
            self.handle_api_error(e)

    async def execute(self, data: object) -> str:
        """Выполняет запрос к LLM в рамках протокола LLMProvider.

        Args:
            data: Данные запроса (строка, список или словарь).

        Returns:
            Строковый ответ модели.
        """
        prompt: str
        if isinstance(data, str):
            prompt = data
        elif isinstance(data, list):
            prompt = "\n".join(str(item) for item in data)
        elif isinstance(data, dict):
            prompt = str(data.get("prompt") or data.get("content") or data.get("messages") or data)
        else:
            prompt = str(data)

        return await self.generate(prompt)

    def pack_single_prompt(self, prompt: str) -> str:
        """Упаковывает строку промпта в формат запроса Gemini.

        Args:
            prompt: Текст промпта.

        Returns:
            Строка промпта.
        """
        return prompt

    async def check_connection(self) -> tuple[bool, str]:
        """Проверяет соединение с API и валидность ключа/модели.

        Returns:
            Кортеж (успех, текстовое сообщение).
        """
        try:
            if hasattr(self._client, "aio") and hasattr(self._client.aio, "models"):
                # Запрашиваем первую страницу списка моделей
                model_iter = await self._client.aio.models.list(config={"page_size": 1})
                if hasattr(model_iter, "__aiter__"):
                    async for _ in model_iter:
                        break
            return True, "Успешно: соединение с Gemini API установлено."
        except Exception as e:
            try:
                self.handle_api_error(e)
            except Exception as parsed_err:
                return False, f"Ошибка Gemini API: {parsed_err}"

    async def close(self) -> None:
        """Асинхронно закрывает ресурсы HTTP-клиента Gemini SDK."""
        try:
            if hasattr(self._client, "aio") and hasattr(self._client.aio, "aclose"):
                await self._client.aio.aclose()
        except Exception as e:
            self.logger.warning("Ошибка при закрытии ресурсов GeminiProvider: %s", e)
