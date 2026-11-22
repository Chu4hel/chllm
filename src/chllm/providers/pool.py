"""Провайдер пула API-ключей с прозрачной ротацией и обработкой лимитов."""

from __future__ import annotations

import asyncio
import logging  # chutils: ignore[ChutilsIntegrationRule]
import time
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from ..exceptions import (
    AuthenticationError,
    NoActiveKeysError,
    ProxyConnectionError,
    QuotaExceededError,
    RateLimitError,
)
from ..orchestrator import LLMProvider

try:
    from chutils.logger import setup_logger
except ImportError:

    def setup_logger(name: str = "chllm.providers.pool") -> Any:
        """Создает резервный логгер, если chutils не установлен.

        Args:
            name: Имя логгера.

        Returns:
            Экземпляр логгера logging.Logger.
        """
        return logging.getLogger(name)


class KeyStatus(StrEnum):
    """Статус API-ключа в пуле."""

    ACTIVE = "active"
    COOLDOWN = "cooldown"
    EXHAUSTED = "exhausted"
    BANNED = "banned"


class RotationStrategy(StrEnum):
    """Стратегия выбора ключа из пула."""

    ROUND_ROBIN = "round_robin"
    LEAST_RECENTLY_USED = "least_recently_used"
    PRIORITY = "priority"


class KeyItem(BaseModel):
    """Информация о состоянии ключа в пуле."""

    key: str
    status: KeyStatus = KeyStatus.ACTIVE
    cooldown_until: float = 0.0
    failure_count: int = 0
    success_count: int = 0
    last_used_at: float = 0.0
    status_message: str = ""

    def is_available(self, now: float | None = None) -> bool:
        """Проверяет доступность ключа для выполнения запросов.

        Args:
            now: Текущая метка времени (по умолчанию time.time()).

        Returns:
            True, если ключ готов к использованию, иначе False.
        """
        current_time = now if now is not None else time.time()
        if self.status == KeyStatus.BANNED:
            return False
        if self.status in (KeyStatus.COOLDOWN, KeyStatus.EXHAUSTED):
            return current_time >= self.cooldown_until
        return self.status == KeyStatus.ACTIVE

    def mark_rate_limited(self, retry_delay: float, message: str = "") -> None:
        """Переводит ключ в состояние кулдауна (429 RPM/TPM).

        Args:
            retry_delay: Задержка в секундах до повторного использования.
            message: Описание причины паузы.
        """
        self.status = KeyStatus.COOLDOWN
        self.cooldown_until = time.time() + max(1.0, retry_delay)
        self.failure_count += 1
        self.status_message = message

    def mark_exhausted(self, duration_seconds: float = 28800.0, message: str = "") -> None:
        """Переводит ключ в состояние исчерпанной суточной квоты.

        Args:
            duration_seconds: Длительность деактивации (по умолчанию 8 часов).
            message: Описание причины.
        """
        self.status = KeyStatus.EXHAUSTED
        self.cooldown_until = time.time() + duration_seconds
        self.failure_count += 1
        self.status_message = message

    def mark_banned(self, message: str = "") -> None:
        """Перманентно блокирует ключ при ошибках аутентификации.

        Args:
            message: Причина блокировки.
        """
        self.status = KeyStatus.BANNED
        self.failure_count += 1
        self.status_message = message

    def mark_success(self) -> None:
        """Фиксирует успешный вызов ключа и сбрасывает счетчик ошибок."""
        self.status = KeyStatus.ACTIVE
        self.failure_count = 0
        self.success_count += 1
        self.last_used_at = time.time()
        self.status_message = ""


class KeyPoolProvider:
    """Провайдер с поддержкой пула ключей и бесшовной ротации.

    Реализует интерфейс LLMProvider. При получении RateLimitError (429) или
    QuotaExceededError прозрачно переключает запрос на следующий доступный ключ
    из пула без прерывания задачи.
    """

    def __init__(
        self,
        keys: Sequence[str] | None = None,
        provider_factory: Callable[[str], LLMProvider] | None = None,
        providers: Sequence[LLMProvider] | None = None,
        strategy: RotationStrategy | str = RotationStrategy.ROUND_ROBIN,
        rate_limit_cooldown: float = 60.0,
        quota_cooldown_hours: float = 8.0,
        max_wait_for_key_seconds: float = 0.0,
        logger: Any | None = None,
    ) -> None:
        """Инициализирует KeyPoolProvider.

        Args:
            keys: Список строковых API-ключей.
            provider_factory: Функция-фабрика создания LLMProvider по ключу.
            providers: Предсозданный список LLMProvider (альтернатива keys + factory).
            strategy: Стратегия ротации ('round_robin', 'least_recently_used', 'priority').
            rate_limit_cooldown: Базовая пауза для 429 RPM/TPM при отсутствии retryDelay (сек).
            quota_cooldown_hours: Время кулдауна при исчерпании суточной квоты (часы).
            max_wait_for_key_seconds: Время ожидания пробуждения ключа, если все ключи в кулдауне.
            logger: Пользовательский логгер.
        """
        self.rate_limit_cooldown = rate_limit_cooldown
        self.quota_cooldown_hours = quota_cooldown_hours
        self.max_wait_for_key_seconds = max_wait_for_key_seconds
        self.logger = logger or setup_logger("chllm.providers.pool")

        if isinstance(strategy, str):
            self.strategy = RotationStrategy(strategy.lower())
        else:
            self.strategy = strategy

        self._keys: list[KeyItem] = []
        self._provider_map: dict[str, LLMProvider] = {}
        self._provider_factory = provider_factory
        self._current_index = 0
        self._lock = asyncio.Lock()

        if providers is not None:
            for idx, prov in enumerate(providers):
                k_id = f"provider-{idx}"
                self._keys.append(KeyItem(key=k_id))
                self._provider_map[k_id] = prov
        elif keys is not None:
            for k in keys:
                self._keys.append(KeyItem(key=k))

        if not self._keys:
            raise ValueError("Необходимо передать хотя бы один ключ или провайдер в KeyPoolProvider.")

    @classmethod
    def from_providers(
        cls,
        providers: Sequence[LLMProvider],
        strategy: RotationStrategy | str = RotationStrategy.ROUND_ROBIN,
        **kwargs: Any,
    ) -> KeyPoolProvider:
        """Создает KeyPoolProvider на основе готовых экземпляров провайдеров.

        Args:
            providers: Последовательность провайдеров.
            strategy: Стратегия ротации.
            **kwargs: Дополнительные аргументы конструктора.

        Returns:
            Сконфигурированный экземпляр KeyPoolProvider.
        """
        return cls(providers=providers, strategy=strategy, **kwargs)

    def _get_provider_for_key(self, key: str) -> LLMProvider:
        """Возвращает или создает экземпляр провайдера для указанного ключа."""
        if key in self._provider_map:
            return self._provider_map[key]

        if self._provider_factory is None:
            raise ValueError(f"Отсутствует provider_factory для создания провайдера под ключ {key[:6]}...")

        provider = self._provider_factory(key)
        self._provider_map[key] = provider
        return provider

    def _refresh_cooldowns(self, now: float) -> None:
        """Переводит остывшие ключи из кулдауна в активное состояние."""
        for item in self._keys:
            if item.status in (KeyStatus.COOLDOWN, KeyStatus.EXHAUSTED) and now >= item.cooldown_until:
                self.logger.info("Ключ %s восстановился после кулдауна.", item.key[:6] + "...")
                item.status = KeyStatus.ACTIVE
                item.cooldown_until = 0.0
                item.status_message = ""

    def _select_key(self, available_keys: list[KeyItem]) -> KeyItem:
        """Выбирает ключ из доступных в соответствии со стратегией ротации."""
        if self.strategy == RotationStrategy.ROUND_ROBIN:
            idx = self._current_index % len(available_keys)
            self._current_index = (self._current_index + 1) % len(available_keys)
            return available_keys[idx]

        if self.strategy == RotationStrategy.LEAST_RECENTLY_USED:
            return min(available_keys, key=lambda k: k.last_used_at)

        # PRIORITY: берем первый доступный по исходному порядку в пуле
        return available_keys[0]

    async def acquire_key(self, excluded_keys: set[str] | None = None) -> tuple[KeyItem, LLMProvider]:
        """Атомарно выбирает активный ключ и возвращает соответствующий провайдер.

        Args:
            excluded_keys: Множество ключей, которые следует исключить (например, уже упавшие в этом запросе).

        Returns:
            Кортеж (KeyItem, LLMProvider).

        Raises:
            NoActiveKeysError: Если нет доступных активных ключей.
        """
        async with self._lock:
            now = time.time()
            self._refresh_cooldowns(now)

            available = [
                k for k in self._keys if k.is_available(now) and (excluded_keys is None or k.key not in excluded_keys)
            ]

            # Если ключей нет, но задан max_wait_for_key_seconds, проверяем спящие
            if not available and self.max_wait_for_key_seconds > 0:
                sleeping = [
                    k
                    for k in self._keys
                    if k.status in (KeyStatus.COOLDOWN, KeyStatus.EXHAUSTED)
                    and (excluded_keys is None or k.key not in excluded_keys)
                ]
                if sleeping:
                    earliest_wake = min(k.cooldown_until for k in sleeping)
                    wait_time = earliest_wake - now
                    if 0 < wait_time <= self.max_wait_for_key_seconds:
                        self.logger.warning("Все ключи в кулдауне. Ожидание пробуждения ближайшего: %.1fс.", wait_time)
                        await asyncio.sleep(wait_time)
                        now = time.time()
                        self._refresh_cooldowns(now)
                        available = [
                            k
                            for k in self._keys
                            if k.is_available(now) and (excluded_keys is None or k.key not in excluded_keys)
                        ]

            if not available:
                banned_count = sum(1 for k in self._keys if k.status == KeyStatus.BANNED)
                cooldown_count = sum(1 for k in self._keys if k.status == KeyStatus.COOLDOWN)
                exhausted_count = sum(1 for k in self._keys if k.status == KeyStatus.EXHAUSTED)
                raise NoActiveKeysError(
                    f"В пуле нет доступных активных ключей (всего: {len(self._keys)}, "
                    f"в кулдауне: {cooldown_count}, исчерпано: {exhausted_count}, заблокировано: {banned_count})."
                )

            chosen = self._select_key(available)
            provider = self._get_provider_for_key(chosen.key)
            return chosen, provider

    async def execute(self, data: object) -> object:
        """Выполняет запрос с автоматической прозрачной ротацией ключей при сбоях.

        Args:
            data: Данные запроса для провайдера.

        Returns:
            Ответ от модели.
        """
        attempted_keys: set[str] = set()
        last_error: Exception | None = None

        while len(attempted_keys) < len(self._keys):
            try:
                key_item, provider = await self.acquire_key(excluded_keys=attempted_keys)
            except NoActiveKeysError as e:
                if last_error is not None:
                    raise last_error from e
                raise

            attempted_keys.add(key_item.key)

            try:
                result = await provider.execute(data)
                key_item.mark_success()
                return result

            except RateLimitError as err:
                last_error = err
                delay = getattr(err, "retry_delay", self.rate_limit_cooldown)
                key_item.mark_rate_limited(delay, message=str(err))
                self.logger.warning(
                    "Ключ %s получил 429 (пауза %.1fс). Ротация на следующий ключ.",
                    key_item.key[:6] + "...",
                    delay,
                )

            except QuotaExceededError as err:
                last_error = err
                key_item.mark_exhausted(self.quota_cooldown_hours * 3600.0, message=str(err))
                self.logger.warning(
                    "Ключ %s исчерпал суточную квоту. Ротация на следующий ключ.",
                    key_item.key[:6] + "...",
                )

            except AuthenticationError as err:
                last_error = err
                key_item.mark_banned(message=str(err))
                self.logger.error("Ключ %s заблокирован (ошибка аутентификации): %s", key_item.key[:6] + "...", err)

            except ProxyConnectionError as err:
                last_error = err
                key_item.mark_rate_limited(10.0, message=str(err))
                self.logger.warning(
                    "Сетевая ошибка на ключе %s: %s. Пробуем следующий ключ.",
                    key_item.key[:6] + "...",
                    err,
                )

            except Exception:
                # Ошибки, не связанные с ключом (например, ContentBlockedError), пробрасываем без ротации
                raise

        if last_error is not None:
            raise last_error

        raise NoActiveKeysError("Все ключи в пуле были опробованы и отклонили запрос.")

    def pack_single_prompt(self, prompt: str) -> object:
        """Делегирует упаковку промпта первому доступному провайдеру или возвращает строку.

        Args:
            prompt: Текст промпта.

        Returns:
            Упакованный запрос.
        """
        for item in self._keys:
            if item.key in self._provider_map:
                return self._provider_map[item.key].pack_single_prompt(prompt)
            if self._provider_factory is not None:
                prov = self._get_provider_for_key(item.key)
                return prov.pack_single_prompt(prompt)
        return prompt

    def get_status_summary(self) -> dict[str, Any]:
        """Возвращает сводку по состоянию всех ключей в пуле.

        Returns:
            Словарь со статистикой активных, спящих и заблокированных ключей.
        """
        now = time.time()
        return {
            "total": len(self._keys),
            "active": sum(1 for k in self._keys if k.is_available(now)),
            "cooldown": sum(1 for k in self._keys if k.status == KeyStatus.COOLDOWN),
            "exhausted": sum(1 for k in self._keys if k.status == KeyStatus.EXHAUSTED),
            "banned": sum(1 for k in self._keys if k.status == KeyStatus.BANNED),
            "keys": [
                {
                    "key_prefix": k.key[:8] + "...",
                    "status": k.status.value,
                    "success": k.success_count,
                    "failures": k.failure_count,
                    "cooldown_remaining": max(0.0, k.cooldown_until - now),
                }
                for k in self._keys
            ],
        }

    async def close(self) -> None:
        """Асинхронно закрывает ресурсы всех созданных провайдеров в пуле."""
        for provider in self._provider_map.values():
            if hasattr(provider, "close") and callable(provider.close):
                try:
                    await provider.close()
                except Exception as e:
                    self.logger.debug("Игнорирование ошибки при закрытии провайдера: %s", e)
