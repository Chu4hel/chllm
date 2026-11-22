"""Файловый буфер задач для взаимодействия с AI-ассистентом (Zero-API режим)."""

import asyncio
import contextlib
import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from chutils.logger import setup_logger
except ImportError:
    import logging

    def setup_logger(name: str = "chllm") -> Any:
        """Создает резервный логгер, если chutils не установлен."""
        return logging.getLogger(name)


logger = setup_logger("chllm.buffer")


def _ensure_dir(path: Path | str) -> Path:
    """Гарантирует существование директории."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write(path: Path | str, content: str, encoding: str = "utf-8") -> None:
    """Атомарно записывает текст в файл через временный файл."""
    target = Path(path)
    _ensure_dir(target.parent)
    dir_path = target.parent
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, prefix=f".tmp_{target.name}_")
    try:
        with open(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
        raise


class FileTaskQueue:
    """Управляет очередью задач в директориях .agent_inbox и .agent_outbox."""

    def __init__(
        self,
        inbox_dir: str | Path = ".agent_inbox",
        outbox_dir: str | Path = ".agent_outbox",
        archive_days: int = 14,
        purge_days: int = 50,
    ) -> None:
        """Инициализирует буфер очереди.

        Args:
            inbox_dir: Директория входящих задач для агента.
            outbox_dir: Директория готовых ответов агента.
            archive_days: Порог дней для перемещения ответов в архив (по умолчанию 14).
            purge_days: Порог дней для окончательного удаления архивных ответов (по умолчанию 50).
        """
        self.inbox_dir = Path(inbox_dir)
        self.outbox_dir = Path(outbox_dir)
        self.archive_dir = self.outbox_dir / "archive"
        self.archive_days = archive_days
        self.purge_days = purge_days

        _ensure_dir(self.inbox_dir)
        _ensure_dir(self.outbox_dir)
        self.cleanup_stale_outbox()

    def clear_response(self, task_id: str) -> None:
        """Удаляет устаревший ответ задачи из outbox и archive.

        Args:
            task_id: Идентификатор задачи.
        """
        (self.outbox_dir / f"{task_id}.json").unlink(missing_ok=True)
        (self.archive_dir / f"{task_id}.json").unlink(missing_ok=True)

    def push_task(
        self,
        task_id: str,
        task_type: str,
        payload: dict[str, Any],
        clear_outbox: bool = False,
    ) -> Path:
        """Записывает задачу в inbox в формате JSON и Markdown.

        Args:
            task_id: Уникальный ID задачи.
            task_type: Тип операции.
            payload: Данные задачи.
            clear_outbox: Удалять ли старый ответ из outbox/archive при постановке задачи.

        Returns:
            Путь к созданному JSON-файлу.
        """
        if clear_outbox:
            self.clear_response(task_id)

        task_data = {
            "task_id": task_id,
            "task_type": task_type,
            "created_at": datetime.now().isoformat(),
            "payload": payload,
        }

        json_path = self.inbox_dir / f"{task_id}.json"
        _atomic_write(
            json_path,
            json.dumps(task_data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        md_path = self.inbox_dir / f"{task_id}.md"
        md_content = self.render_task_markdown(task_id, task_type, payload)
        _atomic_write(md_path, md_content, encoding="utf-8")

        logger.info(
            "Задача %s (%s) отправлена в буфер агента: %s",
            task_id,
            task_type,
            json_path,
        )
        return json_path

    def has_response(self, task_id: str) -> bool:
        """Проверяет, появился ли ответ в outbox или архиве.

        Args:
            task_id: ID задачи.

        Returns:
            True, если ответ готов.
        """
        resp_file = self.outbox_dir / f"{task_id}.json"
        archive_file = self.archive_dir / f"{task_id}.json"
        return resp_file.exists() or archive_file.exists()

    def find_latest_response_by_prefix(self, prefix: str) -> dict[str, Any] | None:
        """Находит самый свежий ответ в outbox, имя файла которого начинается с prefix.

        Args:
            prefix: Префикс имени задачи.

        Returns:
            Словарь с ответом агента или None.
        """
        latest_inbox_mtime = 0.0
        if self.inbox_dir.exists():
            inbox_candidates = list(self.inbox_dir.glob(f"{prefix}.json")) + list(
                self.inbox_dir.glob(f"{prefix}_*.json")
            )
            for p in inbox_candidates:
                try:
                    m = p.stat().st_mtime
                    latest_inbox_mtime = max(latest_inbox_mtime, m)
                except OSError:
                    pass

        candidates = list(self.outbox_dir.glob(f"{prefix}.json")) + list(self.outbox_dir.glob(f"{prefix}_*.json"))
        if not candidates:
            candidates = list(self.archive_dir.glob(f"{prefix}.json")) + list(self.archive_dir.glob(f"{prefix}_*.json"))
        if not candidates:
            return None

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        top = candidates[0]
        try:
            if latest_inbox_mtime > 0 and top.stat().st_mtime < latest_inbox_mtime:
                return None
            return json.loads(top.read_text(encoding="utf-8"))
        except Exception:
            return None

    def get_response(self, task_id: str, cleanup: bool = False) -> dict[str, Any] | None:
        """Считывает ответ из outbox или архива.

        Args:
            task_id: ID задачи.
            cleanup: Удалять ли файлы задачи после прочтения.

        Returns:
            Словарь с ответом агента или None.
        """
        resp_file = self.outbox_dir / f"{task_id}.json"
        if not resp_file.exists():
            archive_file = self.archive_dir / f"{task_id}.json"
            if archive_file.exists():
                resp_file = archive_file
            else:
                return None

        try:
            with open(resp_file, encoding="utf-8") as f:
                data = json.load(f)

            if cleanup:
                try:
                    resp_file.unlink(missing_ok=True)
                except OSError as e:
                    logger.debug("Не удалось удалить файл ответа %s: %s", resp_file, e)

            self._cleanup_task_inbox(task_id)
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Ошибка чтения результата %s: %s. Ожидание завершения записи...",
                task_id,
                e,
            )
            return None

    def get_result(self, task_id: str) -> dict[str, Any] | None:
        """Получает готовый результат задачи из outbox.

        Args:
            task_id: Уникальный ID задачи.

        Returns:
            Словарь с результатом или None.
        """
        return self.get_response(task_id, cleanup=False)

    async def wait_for_response(
        self,
        task_id: str,
        timeout_seconds: float = 300.0,
        check_interval: float = 1.0,
        cleanup: bool = False,
    ) -> dict[str, Any] | None:
        """Ожидает появления ответа в outbox в асинхронном цикле.

        Args:
            task_id: ID задачи.
            timeout_seconds: Максимальное время ожидания в секундах.
            check_interval: Интервал проверки в секундах.
            cleanup: Удалять ли задачу после получения.

        Returns:
            Ответ задачи или None при таймауте.
        """
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout_seconds:
            if self.has_response(task_id):
                return self.get_response(task_id, cleanup=cleanup)
            await asyncio.sleep(check_interval)
        return None

    async def wait_for_result(
        self,
        task_id: str,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> dict[str, Any]:
        """Асинхронно ожидает появления результата в outbox с выбросом TimeoutError.

        Args:
            task_id: Уникальный ID задачи.
            timeout: Максимальное время ожидания в секундах.
            poll_interval: Интервал опроса в секундах.

        Returns:
            Словарь с результатом ответа агента.

        Raises:
            TimeoutError: Если ответ не появился за отведенное время.
        """
        res = await self.wait_for_response(
            task_id=task_id,
            timeout_seconds=timeout,
            check_interval=poll_interval,
            cleanup=False,
        )
        if res is not None:
            return res

        raise TimeoutError(f"Превышен таймаут ожидания ответа агента для задачи {task_id} ({timeout}с)")

    def list_pending_tasks(self) -> list[str]:
        """Возвращает список ID задач, ожидающих обработки в inbox.

        Returns:
            Список строковых идентификаторов задач.
        """
        if not self.inbox_dir.exists():
            return []
        task_ids: list[str] = []
        for p in self.inbox_dir.glob("*.json"):
            if not p.name.startswith("."):
                task_ids.append(p.stem)
        return sorted(task_ids)

    def has_pending_tasks(self) -> bool:
        """Проверяет, есть ли необработанные задачи в inbox.

        Returns:
            True, если есть задачи в очереди, иначе False.
        """
        return len(self.list_pending_tasks()) > 0

    def list_inbox_task_ids(self) -> list[str]:
        """Возвращает список всех ID задач во входящей очереди.

        Returns:
            Список строковых идентификаторов задач.
        """
        return self.list_pending_tasks()

    def list_outbox_task_ids(self) -> list[str]:
        """Возвращает список ID задач, для которых готов ответ в outbox.

        Returns:
            Список строковых идентификаторов задач с ответами.
        """
        if not self.outbox_dir.exists():
            return []
        task_ids: list[str] = []
        for p in self.outbox_dir.glob("*.json"):
            if not p.name.startswith("."):
                task_ids.append(p.stem)
        return sorted(task_ids)

    def cleanup_stale_outbox(
        self,
        archive_days: int | None = None,
        purge_days: int | None = None,
    ) -> dict[str, int]:
        """Очищает устаревшие файлы в outbox: архивирует старые и удаляет просроченные.

        Args:
            archive_days: Порог дней для архивации (по умолчанию self.archive_days).
            purge_days: Порог дней для безвозвратного удаления (по умолчанию self.purge_days).

        Returns:
            Словарь со счетчиками {'archived': count, 'purged': count}.
        """
        arch_limit = archive_days if archive_days is not None else self.archive_days
        purge_limit = purge_days if purge_days is not None else self.purge_days

        counts = {"archived": 0, "purged": 0}
        if not self.outbox_dir.exists():
            return counts

        now = time.time()
        archive_threshold = now - (arch_limit * 86400)
        purge_threshold = now - (purge_limit * 86400)

        for item in self.outbox_dir.iterdir():
            if item.is_file() and not item.name.startswith("."):
                try:
                    mtime = item.stat().st_mtime
                    if mtime < archive_threshold:
                        _ensure_dir(self.archive_dir)
                        shutil.move(str(item), str(self.archive_dir / item.name))
                        counts["archived"] += 1
                        logger.info("Файл %s перемещен в архив outbox", item.name)
                except OSError as e:
                    logger.warning("Не удалось архивировать файл %s: %s", item.name, e)

        if self.archive_dir.exists():
            for item in self.archive_dir.iterdir():
                if item.is_file():
                    try:
                        mtime = item.stat().st_mtime
                        if mtime < purge_threshold:
                            item.unlink()
                            counts["purged"] += 1
                            logger.info(
                                "Файл %s окончательно удален из архива по таймауту",
                                item.name,
                            )
                    except OSError as e:
                        logger.warning("Не удалось удалить файл %s: %s", item.name, e)

        return counts

    def _cleanup_task_inbox(self, task_id: str) -> None:
        """Удаляет обработанные файлы задачи из inbox."""
        for ext in (".json", ".md"):
            p = self.inbox_dir / f"{task_id}{ext}"
            if p.exists():
                try:
                    p.unlink()
                except OSError as e:
                    logger.debug("Не удалось удалить файл %s: %s", p, e)

    def render_task_markdown(
        self,
        task_id: str,
        task_type: str,
        payload: dict[str, Any],
    ) -> str:
        """Генерирует Markdown-представление задачи для удобного чтения AI-ассистентом.

        Args:
            task_id: Уникальный ID задачи.
            task_type: Тип задачи.
            payload: Данные задачи.

        Returns:
            Текст в формате Markdown.
        """
        lines = [
            f"# Задача для AI-Ассистента: {task_type}",
            f"- **ID задачи:** `{task_id}`",
            f"- **Создано:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Инструкция по выполнению",
            "1. Проанализируйте данные в разделе **Входные данные**.",
            f"2. Сохраните ответ строго в файл `.agent_outbox/{task_id}.json`.",
            "3. Ответ должен быть валидным JSON со структурой, описанной в схеме.",
            "",
            "## Входные данные (Payload)",
            "```json",
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            "```",
        ]
        return "\n".join(lines)
