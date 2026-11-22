"""Тесты для KeyPoolProvider, KeyItem и стратегий ротации ключей."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from chllm import (
    AuthenticationError,
    ContentBlockedError,
    KeyItem,
    KeyPoolProvider,
    KeyStatus,
    NoActiveKeysError,
    QuotaExceededError,
    RateLimitError,
    RotationStrategy,
)


def test_key_item_lifecycle():
    """Проверка смены состояний KeyItem при ошибках и успехе."""
    item = KeyItem(key="sk-test-1")
    assert item.is_available() is True

    # 429 RateLimit
    item.mark_rate_limited(retry_delay=30.0, message="RPM exceeded")
    assert item.status == KeyStatus.COOLDOWN
    assert item.cooldown_until > time.time()
    assert item.is_available() is False
    assert item.failure_count == 1

    # Симуляция выхода из кулдауна
    assert item.is_available(now=item.cooldown_until + 1.0) is True

    # Успех
    item.mark_success()
    assert item.status == KeyStatus.ACTIVE
    assert item.failure_count == 0
    assert item.success_count == 1

    # Суточная квота
    item.mark_exhausted(duration_seconds=3600.0)
    assert item.status == KeyStatus.EXHAUSTED
    assert item.is_available() is False

    # Бан
    item.mark_banned(message="Invalid API Key")
    assert item.status == KeyStatus.BANNED
    assert item.is_available() is False
    assert item.is_available(now=time.time() + 1000000) is False


@pytest.mark.asyncio
async def test_round_robin_rotation():
    """Проверка циклического переключения ключей по стратегии Round-Robin."""
    calls: list[str] = []

    def mock_factory(key: str):
        prov = MagicMock()

        async def _exec(data):
            calls.append(key)
            return f"Answer from {key}"

        prov.execute = AsyncMock(side_effect=_exec)
        prov.pack_single_prompt = lambda p: p
        return prov

    pool = KeyPoolProvider(
        keys=["key-A", "key-B", "key-C"],
        provider_factory=mock_factory,
        strategy=RotationStrategy.ROUND_ROBIN,
    )

    res1 = await pool.execute("q1")
    res2 = await pool.execute("q2")
    res3 = await pool.execute("q3")
    res4 = await pool.execute("q4")

    assert calls == ["key-A", "key-B", "key-C", "key-A"]
    assert res1 == "Answer from key-A"
    assert res2 == "Answer from key-B"
    assert res3 == "Answer from key-C"
    assert res4 == "Answer from key-A"


@pytest.mark.asyncio
async def test_least_recently_used_rotation():
    """Проверка стратегии Least-Recently-Used (выбор самого старого ключа)."""
    calls: list[str] = []

    def mock_factory(key: str):
        prov = MagicMock()

        async def _exec(data):
            calls.append(key)
            return f"Answer from {key}"

        prov.execute = AsyncMock(side_effect=_exec)
        prov.pack_single_prompt = lambda p: p
        return prov

    pool = KeyPoolProvider(
        keys=["key-1", "key-2"],
        provider_factory=mock_factory,
        strategy=RotationStrategy.LEAST_RECENTLY_USED,
    )

    await pool.execute("1")
    await pool.execute("2")
    await pool.execute("3")

    assert calls == ["key-1", "key-2", "key-1"]


@pytest.mark.asyncio
async def test_priority_rotation():
    """Проверка стратегии Priority (использование первого доступного ключа)."""
    calls: list[str] = []

    def mock_factory(key: str):
        prov = MagicMock()

        async def _exec(data):
            calls.append(key)
            return f"Answer from {key}"

        prov.execute = AsyncMock(side_effect=_exec)
        prov.pack_single_prompt = lambda p: p
        return prov

    pool = KeyPoolProvider(
        keys=["key-primary", "key-secondary"],
        provider_factory=mock_factory,
        strategy=RotationStrategy.PRIORITY,
    )

    await pool.execute("1")
    await pool.execute("2")

    # Основной ключ успешен, поэтому второй не вызывается
    assert calls == ["key-primary", "key-primary"]


@pytest.mark.asyncio
async def test_transparent_failover_on_rate_limit():
    """Проверка немедленного бесшовного переключения на следующий ключ при 429."""
    prov_a = MagicMock()
    prov_a.execute = AsyncMock(side_effect=RateLimitError("429 Too Many Requests", retry_delay=60.0))
    prov_a.pack_single_prompt = lambda p: p

    prov_b = MagicMock()
    prov_b.execute = AsyncMock(return_value="Answer from B")
    prov_b.pack_single_prompt = lambda p: p

    pool = KeyPoolProvider.from_providers(
        providers=[prov_a, prov_b],
        strategy=RotationStrategy.ROUND_ROBIN,
    )

    # Запрос должен завершиться успехом на prov_b, несмотря на ошибку prov_a
    result = await pool.execute("Test prompt")
    assert result == "Answer from B"

    status = pool.get_status_summary()
    assert status["total"] == 2
    assert status["active"] == 1
    assert status["cooldown"] == 1

    # Следующий запрос сразу пойдет на prov_b, так как prov_a в кулдауне
    result2 = await pool.execute("Test prompt 2")
    assert result2 == "Answer from B"
    assert prov_b.execute.call_count == 2


@pytest.mark.asyncio
async def test_transparent_failover_on_daily_quota():
    """Проверка ротации ключа при исчерпании суточной квоты."""
    prov_1 = MagicMock()
    prov_1.execute = AsyncMock(side_effect=QuotaExceededError("Daily limit reached"))
    prov_1.pack_single_prompt = lambda p: p

    prov_2 = MagicMock()
    prov_2.execute = AsyncMock(return_value="Success from 2")
    prov_2.pack_single_prompt = lambda p: p

    pool = KeyPoolProvider.from_providers(
        providers=[prov_1, prov_2],
        quota_cooldown_hours=8.0,
    )

    result = await pool.execute("Hello")
    assert result == "Success from 2"

    status = pool.get_status_summary()
    assert status["exhausted"] == 1
    assert status["active"] == 1


@pytest.mark.asyncio
async def test_permanent_ban_on_auth_error():
    """Проверка перманентной блокировки неверного ключа."""
    prov_bad = MagicMock()
    prov_bad.execute = AsyncMock(side_effect=AuthenticationError("Invalid API Key"))
    prov_bad.pack_single_prompt = lambda p: p

    prov_good = MagicMock()
    prov_good.execute = AsyncMock(return_value="Good response")
    prov_good.pack_single_prompt = lambda p: p

    pool = KeyPoolProvider.from_providers(providers=[prov_bad, prov_good])

    result = await pool.execute("Query")
    assert result == "Good response"

    status = pool.get_status_summary()
    assert status["banned"] == 1
    assert status["active"] == 1


@pytest.mark.asyncio
async def test_non_recoverable_error_does_not_rotate():
    """Проверка, что ошибки контента пробрасываются немедленно без ротации."""
    prov_1 = MagicMock()
    prov_1.execute = AsyncMock(side_effect=ContentBlockedError("Safety filter"))
    prov_1.pack_single_prompt = lambda p: p

    prov_2 = MagicMock()
    prov_2.execute = AsyncMock(return_value="Should not be called")
    prov_2.pack_single_prompt = lambda p: p

    pool = KeyPoolProvider.from_providers(providers=[prov_1, prov_2])

    with pytest.raises(ContentBlockedError):
        await pool.execute("Dangerous query")

    assert prov_1.execute.call_count == 1
    assert prov_2.execute.call_count == 0


@pytest.mark.asyncio
async def test_all_keys_exhausted_raises_error():
    """Проверка генерации NoActiveKeysError, когда все ключи исчерпаны."""
    prov_1 = MagicMock()
    prov_1.execute = AsyncMock(side_effect=RateLimitError("429 Limit", retry_delay=100.0))
    prov_1.pack_single_prompt = lambda p: p

    prov_2 = MagicMock()
    prov_2.execute = AsyncMock(side_effect=RateLimitError("429 Limit 2", retry_delay=100.0))
    prov_2.pack_single_prompt = lambda p: p

    pool = KeyPoolProvider.from_providers(providers=[prov_1, prov_2])

    with pytest.raises(RateLimitError):
        await pool.execute("Query")

    # Теперь все ключи в кулдауне, следующий вызов бросает NoActiveKeysError
    with pytest.raises(NoActiveKeysError):
        await pool.execute("Query 2")


@pytest.mark.asyncio
async def test_close_and_pack_prompt():
    """Проверка делегирования pack_single_prompt и закрытия провайдеров."""
    prov = MagicMock()
    prov.pack_single_prompt = MagicMock(return_value={"custom": "packed"})
    prov.close = AsyncMock()

    pool = KeyPoolProvider.from_providers(providers=[prov])
    packed = pool.pack_single_prompt("Test prompt")
    assert packed == {"custom": "packed"}

    await pool.close()
    prov.close.assert_called_once()
