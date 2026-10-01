"""Провайдер для работы с OpenAI-совместимыми шлюзами и локальными роутерами.

Поддерживает прямые HTTP-вызовы через httpx.AsyncClient к Browser LLM Gateway (BLG),
vLLM, LiteLLM, Ollama, LocalAI и удаленным OpenAI-совместимым API.
"""

from __future__ import annotations

import asyncio
import logging  # chutils: ignore[ChutilsIntegrationRule]
import re
import urllib.parse
import uuid
from typing import Any, NoReturn

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
from ..metrics import UsageMetrics
from ..utils import mask_proxy_url

try:
    from chutils.logger import setup_logger
except ImportError:

    def setup_logger(name: str = "chllm.providers.router") -> Any:
        """Создает резервный логгер, если chutils не установлен.

        Args:
            name: Имя логгера.

        Returns:
            Экземпляр логгера logging.Logger.
        """
        return logging.getLogger(name)


class OpenAIRouterProvider:
    """Провайдер для работы с OpenAI-совместимыми роутерами и локальными шлюзами через httpx.

    Реализует управление сессиями BLG (Browser LLM Gateway), пулы соединений,
    извлечение метрик UsageMetrics и интеллектуальный разбор кодов ошибок HTTP.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        proxy_url: str | None = None,
        is_blg: bool | None = None,
        session_id: str | None = None,
        client: Any | None = None,
        timeout: float = 600.0,
        temperature: float | None = None,
        max_tokens: int = 65536,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_payload: dict[str, Any] | None = None,
        logger: Any | None = None,
    ) -> None:
        """Инициализирует OpenAIRouterProvider.

        Args:
            base_url: Базовый URL роутера (например, 'http://127.0.0.1:8000/v1').
            model: Имя модели (например, 'gpt-4o-mini', 'gemini-2.5-flash').
            api_key: API ключ (если не указан для локальных шлюзов, используется 'sk-dummy').
            proxy_url: URL прокси-сервера.
            is_blg: Флаг работы с Browser LLM Gateway (если None, определяется автоматически).
            session_id: Пользовательский фиксированный ID сессии.
            client: Экземпляр httpx.AsyncClient (если уже создан).
            timeout: Таймаут запросов в секундах.
            temperature: Температура генерации.
            max_tokens: Максимальное количество токенов в ответе.
            thinking_level: Уровень размышлений (например, 'low', 'medium', 'high', 'off').
            thinking_budget: Бюджет токенов на размышления.
            extra_headers: Дополнительные HTTP-заголовки.
            extra_payload: Дополнительные поля тела запроса.
            logger: Пользовательский логгер.
        """
        self.base_url = base_url.strip().rstrip("/")
        self.model = model
        self.api_key = api_key or "sk-dummy"
        self.proxy_url = proxy_url
        self.custom_session_id = session_id
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_level = thinking_level
        self.thinking_budget = thinking_budget
        self.extra_headers = extra_headers or {}
        self.extra_payload = extra_payload or {}
        self.logger = logger or setup_logger("chllm.providers.router")
        self.last_usage: UsageMetrics | None = None

        if is_blg is not None:
            self.is_blg = is_blg
        else:
            url_lower = self.base_url.lower()
            self.is_blg = bool(
                "blg" in url_lower
                or "bridge" in url_lower
                or "router" in url_lower
                or "127.0.0.1:8000" in url_lower
                or "localhost:8000" in url_lower
            )

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = self._init_client()
            self._owns_client = True

    def _init_client(self) -> Any:
        """Инициализирует клиент httpx.AsyncClient с настройками пула и прокси."""
        try:
            import httpx
        except ImportError as err:
            raise ImportError(
                "Для использования OpenAIRouterProvider необходимо установить пакет httpx: "
                "pip install 'chllm[router]' или pip install httpx"
            ) from err

        client_kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(self.timeout, connect=60.0, read=self.timeout, write=180.0, pool=60.0),
            "limits": httpx.Limits(max_keepalive_connections=20, max_connections=50),
        }
        if self.proxy_url:
            masked = mask_proxy_url(self.proxy_url)
            self.logger.info("Используется прокси для OpenAIRouterProvider: %s", masked)
            client_kwargs["proxy"] = self.proxy_url

        return httpx.AsyncClient(**client_kwargs)

    def _create_request_session_id(self) -> str:
        """Создает уникальный ASCII-безопасный идентификатор сессии.

        Returns:
            Уникальная строка идентификатора сессии.
        """
        if self.custom_session_id:
            return urllib.parse.quote(self.custom_session_id, safe="-_.~")
        raw_id = f"chllm-{uuid.uuid4().hex[:12]}"
        return urllib.parse.quote(raw_id, safe="-_.~")

    def _get_request_headers(self, session_id: str | None = None) -> dict[str, str]:
        """Формирует HTTP-заголовки запроса с учетом сессионных меток BLG.

        Args:
            session_id: Идентификатор сессии для изоляции запроса.

        Returns:
            Словарь заголовков HTTP.
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        sid = session_id or self._create_request_session_id()

        if self.is_blg:
            headers["X-Input-Mode"] = "paste"
            headers["X-Paste-Threshold"] = "0"
            headers["X-Temporary-Chat"] = "true"
            headers["X-Session-ID"] = sid
            headers["X-API-Key"] = sid
            headers["Authorization"] = f"Bearer {sid}"
        else:
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            if sid:
                headers["X-Session-ID"] = sid

        headers.update(self.extra_headers)
        return headers

    def _get_chat_url(self) -> str:
        """Возвращает полный URL для эндпоинта chat completions.

        Returns:
            Строка полного URL эндпоинта.
        """
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    async def _close_session(self, session_id: str) -> None:
        """Удаляет временную сессию в шлюзе BLG и закрывает связанный контекст браузера.

        Args:
            session_id: Идентификатор сессии.
        """
        if not session_id or not self.is_blg:
            return
        if getattr(self._client, "is_closed", False):
            return

        sessions_url = (
            f"{self.base_url}/sessions/{session_id}"
            if self.base_url.endswith("/v1")
            else f"{self.base_url}/v1/sessions/{session_id}"
        )
        try:
            delete_headers = self._get_request_headers(session_id)
            await self._client.delete(sessions_url, headers=delete_headers, timeout=5.0)
        except Exception as e:
            self.logger.debug("Игнорирование ошибки при закрытии сессии BLG %s: %s", session_id, e)

    def handle_api_error(self, e: BaseException) -> NoReturn:
        """Разбирает ошибки сети, прокси и HTTP-статусов роутера.

        Args:
            e: Исходное исключение.

        Raises:
            ProxyConnectionError: При ошибках сети, прокси или таймаутах.
            RateLimitError: При коде 429 (RPM/TPM).
            QuotaExceededError: При исчерпании суточной квоты.
            ServiceUnavailableError: При ошибках 500, 502, 503, 504.
            AuthenticationError: При кодах 401, 403 или требовании авторизации.
            ContentBlockedError: При цензуре контента.
            LocationUnsupportedError: При гео-блокировках.
            CHLLMError: При остальных ошибках.

        Returns:
            Ничего не возвращает (всегда возбуждает типизированное исключение NoReturn).
        """
        # 1. Если это таймаут или прерывание запроса
        if isinstance(e, (asyncio.TimeoutError, TimeoutError, asyncio.CancelledError)):
            raise ProxyConnectionError(f"Таймаут или прерывание запроса к роутеру: {type(e).__name__}") from e

        # 2. Если уже типизированная ошибка библиотеки
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

        # 3. Разворачивание причин
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

        # 4. Сетевые ошибки httpx и сокетов
        net_signatures = (
            "readerror",
            "timeoutexception",
            "connecterror",
            "proxyerror",
            "networkerror",
            "remoteprotocolerror",
            "connectionreseterror",
            "connectionrefusederror",
        )
        if (
            any(sig in error_type_name for sig in net_signatures)
            or any(sig in root_type_name for sig in net_signatures)
            or "server disconnected without sending a response" in combined_text
        ):
            msg = f"Ошибка связи с роутером/сервером: {type(root_error).__name__} - {root_error!s}"
            raise ProxyConnectionError(msg) from e

        # 5. Обработка HTTPStatusError от httpx
        resp_obj = getattr(e, "response", None)
        status_code = getattr(resp_obj, "status_code", None)
        response_text = ""
        if resp_obj is not None and hasattr(resp_obj, "text"):
            response_text = str(resp_obj.text)
        resp_lower = response_text.lower()
        full_text = f"{combined_text} {resp_lower}"

        # 5.1 Геолокация
        if "user location is not supported" in full_text:
            raise LocationUnsupportedError(f"Заблокировано по геолокации роутера: {response_text}") from e

        # 5.2 Ошибки авторизации (401, 403, требование авторизации)
        if status_code in (401, 403) or "требуют авторизации" in full_text or "unauthorized" in full_text:
            raise AuthenticationError(f"Ошибка авторизации роутера ({status_code}): {response_text}") from e

        # 5.3 Недоступность сервера / шлюза (500, 502, 503, 504)
        if status_code in (500, 502, 503, 504) or any(
            kw in full_text for kw in ["bad gateway", "gateway timeout", "service unavailable", "overloaded"]
        ):
            retry_delay = 30.0
            match = re.search(r'["\']?retryDelay["\']?:\s*["\']?(\d+(?:\.\d+)?)s?["\']?', full_text, re.IGNORECASE)
            if match:
                try:
                    retry_delay = float(match.group(1))
                except ValueError:
                    retry_delay = 30.0
            raise ServiceUnavailableError(
                f"Роутер/шлюз временно недоступен ({status_code}): {response_text}. Повтор через {retry_delay}с.",
                retry_delay=retry_delay,
            ) from e

        # 5.4 Превышение лимитов (429 / RPM vs Суточная квота)
        retry_delay_match = re.search(
            r'["\']?retryDelay["\']?:\s*["\']?(\d+(?:\.\d+)?)s?["\']?', full_text, re.IGNORECASE
        )
        quota_keywords = (
            "per day",
            "daily quota",
            "day quota",
            "исчерпали квоту",
            "исчерпана квота",
            "исчерпан",
            "quota exceeded",
            "free tier quota exhausted",
        )
        is_daily_quota = (
            any(kw in full_text for kw in quota_keywords)
            and retry_delay_match is None
            and "rate limit" not in full_text
        )

        if is_daily_quota:
            raise QuotaExceededError(f"Исчерпана суточная квота роутера: {response_text}") from e

        if status_code == 429 or "rate limit" in full_text or "too many requests" in full_text:
            retry_delay = 60.0
            if retry_delay_match:
                try:
                    retry_delay = float(retry_delay_match.group(1))
                except ValueError:
                    retry_delay = 60.0
            raise RateLimitError(
                f"Превышен временный лимит запросов роутера (429). Пауза {retry_delay}с.",
                retry_delay=retry_delay,
            ) from e

        # 5.5 Блокировка контента
        if "content_filter" in full_text or "safety" in full_text or "blocked" in full_text:
            raise ContentBlockedError(f"Запрос заблокирован фильтром безопасности: {response_text}") from e

        if status_code:
            raise CHLLMError(f"HTTP ошибка роутера ({status_code}): {response_text}") from e

        raise CHLLMError(f"Неизвестная ошибка роутера: {type(e).__name__} - {e}") from e

    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
    ) -> str:
        """Выполняет запрос генерации текста через OpenAI-совместимый эндпоинт роутера.

        Args:
            prompt: Текст пользовательского запроса.
            system_instruction: Опциональная системная инструкция.
            thinking_level: Уровень размышлений (например, 'medium', 'low', 'off').
            thinking_budget: Бюджет токенов на размышления.

        Returns:
            Строковый ответ модели.
        """
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        session_id = self._create_request_session_id()

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        # Специфика параметров BLG
        if self.is_blg:
            payload.update(
                {
                    "input_mode": "paste",
                    "paste_threshold": 0,
                    "temporary_chat": True,
                    "session_id": session_id,
                    "user": session_id,
                    "new_chat": True,
                    "generation_timeout": self.timeout,
                }
            )
        else:
            payload["user"] = session_id

        # Настройка Thinking / Reasoning параметров
        eff_level = thinking_level or self.thinking_level
        eff_budget = thinking_budget if thinking_budget is not None else self.thinking_budget

        if eff_level:
            payload["reasoning_effort"] = str(eff_level).lower()
            if self.is_blg:
                payload["thinking_level"] = str(eff_level).lower()
        elif eff_budget is not None and eff_budget > 0:
            payload["thinking_budget"] = eff_budget

        payload.update(self.extra_payload)
        headers = self._get_request_headers(session_id)

        try:
            response = await self._client.post(
                self._get_chat_url(),
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Извлечение метрик расхода токенов
            usage_data = data.get("usage")
            if usage_data and isinstance(usage_data, dict):
                self.last_usage = UsageMetrics(
                    prompt_tokens=usage_data.get("prompt_tokens", 0) or 0,
                    completion_tokens=usage_data.get("completion_tokens", 0) or 0,
                )

            choices = data.get("choices", [])
            if not choices:
                raise ContentBlockedError("Ответ от роутера пустой: отсутствует список choices.")

            message = choices[0].get("message", {})
            content = message.get("content", "")
            if not content:
                raise ContentBlockedError("Ответ от роутера содержит пустое поле content.")

            return str(content)

        except Exception as e:
            self.handle_api_error(e)
        finally:
            if self.is_blg:
                await self._close_session(session_id)

    async def execute(self, data: object) -> str:
        """Выполняет запрос к LLM в рамках протокола LLMProvider.

        Args:
            data: Строка, список сообщений или словарь с параметрами запроса.

        Returns:
            Строковый ответ модели.
        """
        if isinstance(data, str):
            return await self.generate(data)
        if isinstance(data, list):
            prompt = "\n".join(str(item) for item in data)
            return await self.generate(prompt)
        if isinstance(data, dict):
            prompt = str(data.get("prompt") or data.get("content") or data.get("messages") or data)
            sys_prompt = data.get("system_instruction") or data.get("system_prompt")
            return await self.generate(prompt, system_instruction=str(sys_prompt) if sys_prompt else None)
        return await self.generate(str(data))

    def pack_single_prompt(self, prompt: str) -> str:
        """Упаковывает строку промпта в формат запроса роутера.

        Args:
            prompt: Текст промпта.

        Returns:
            Строка промпта.
        """
        return prompt

    async def check_connection(self) -> tuple[bool, str]:
        """Проверяет доступность эндпоинта роутера или моделей.

        Returns:
            Кортеж (успех, текстовое сообщение).
        """
        models_url = f"{self.base_url}/models" if self.base_url.endswith("/v1") else f"{self.base_url}/v1/models"
        headers = self._get_request_headers()

        # 1. Попытка запросить /models
        try:
            resp = await self._client.get(models_url, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                return True, f"Успешно: Соединение с роутером {self.base_url} установлено (/models OK)."
        except Exception:
            pass

        # 2. Тестовый пинг-запрос
        try:
            result = await self.generate("ping")
            if result:
                return True, f"Успешно: Тестовый запрос к роутеру {self.base_url} выполнен."
            return False, f"Ответ от роутера {self.base_url} пуст."
        except Exception as e:
            try:
                self.handle_api_error(e)
            except Exception as parsed:
                return False, f"Ошибка подключения к роутеру {self.base_url}: {parsed}"

    async def close(self) -> None:
        """Асинхронно освобождает ресурсы HTTP-клиента."""
        if self._owns_client and hasattr(self._client, "aclose") and not self._client.is_closed:
            try:
                await self._client.aclose()
            except Exception as e:
                self.logger.warning("Ошибка при закрытии клиента OpenAIRouterProvider: %s", e)
