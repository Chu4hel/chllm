#!/usr/bin/env python3
"""Удаляет подпись Codebuff из сообщений незапушенных коммитов.

Скрипт переписывает только коммиты из диапазона ``@{upstream}..HEAD``,
сохраняя деревья, авторов, даты и порядок. Перед изменением создаётся
резервная ссылка ``refs/backup/codebuff-clean/<timestamp>``.

По умолчанию выполняется только предпросмотр. Для переписывания истории
нужно передать ``--apply``. После этого отправка потребует force push:

    git push --force-with-lease
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import NoReturn

from chutils.time import utc_now

DEFAULT_RANGE = "@{upstream}..HEAD"
FOOTER_PATTERNS = (
    re.compile(r"^(?:🤖\s*)?Generated with Codebuff(?:\s*🤖)?$", re.IGNORECASE),
    re.compile(
        r"^Co-Authored-By:\s*Codebuff\s*<noreply@codebuff\.com>$",
        re.IGNORECASE,
    ),
)
IDENTITY_RE = re.compile(r"^(.+) <(.+)> (\d+) ([+-]\d{4})$")


@dataclass(frozen=True)
class Identity:
    """Автор или коммиттер из заголовка git-объекта."""

    name: str
    email: str
    date: str


@dataclass(frozen=True)
class CommitData:
    """Данные коммита, необходимые для его пересоздания."""

    tree: str
    parents: tuple[str, ...]
    author: Identity
    committer: Identity
    message: str


class GitError(RuntimeError):
    """Ошибка выполнения git-команды."""


def run_git(
    *args: str,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> str:
    """Запускает git и возвращает stdout.

    Args:
        *args: Аргументы git без самого имени команды.
        input_text: Данные, передаваемые в stdin.
        env: Дополнительные переменные окружения.
        check: Выбрасывать ли GitError при ненулевом коде возврата.

    Returns:
        Текст stdout выполненной команды.

    Raises:
        GitError: Если git завершился с ошибкой и ``check`` включён.
    """
    completed = subprocess.run(
        ["git", *args],
        input=input_text,
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=env,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GitError(f"git {' '.join(args)}: {detail}")
    return completed.stdout


def is_footer_line(line: str) -> bool:
    """Проверяет, является ли строка подписью Codebuff.

    Args:
        line: Проверяемая строка.

    Returns:
        ``True``, если строка является подписью Codebuff.
    """
    return any(pattern.fullmatch(line.strip()) for pattern in FOOTER_PATTERNS)


def strip_footer(message: str) -> tuple[str, bool]:
    """Удаляет подпись Codebuff в конце сообщения коммита.

    Args:
        message: Исходное сообщение коммита.

    Returns:
        Кортеж из очищенного сообщения и признака изменения.
    """
    lines = message.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()

    changed = False
    while lines and is_footer_line(lines[-1]):
        changed = True
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()

    if not changed:
        return message, False
    return "\n".join(lines) + ("\n" if lines else ""), True


def parse_identity(value: str) -> Identity:
    """Разбирает строку author/committer из git-объекта.

    Args:
        value: Строка вида ``Имя <email> timestamp timezone``.

    Returns:
        Разобранная идентичность.

    Raises:
        GitError: Если формат строки не распознан.
    """
    match = IDENTITY_RE.fullmatch(value)
    if match is None:
        raise GitError(f"Не удалось разобрать идентичность: {value}")
    name, email, timestamp, timezone_offset = match.groups()
    return Identity(name=name, email=email, date=f"{timestamp} {timezone_offset}")


def read_commit(sha: str) -> CommitData:
    """Читает дерево, родителей, авторов и сообщение коммита.

    Args:
        sha: Полный хеш коммита.

    Returns:
        Данные коммита.

    Raises:
        GitError: Если коммит подписан или имеет неожиданный формат.
    """
    raw = run_git("cat-file", "commit", sha)
    header, _, message = raw.partition("\n\n")
    tree = ""
    parents: list[str] = []
    author: Identity | None = None
    committer: Identity | None = None

    for line in header.splitlines():
        if line.startswith(" "):
            continue
        if line.startswith("tree "):
            tree = line.removeprefix("tree ")
        elif line.startswith("parent "):
            parents.append(line.removeprefix("parent "))
        elif line.startswith("author "):
            author = parse_identity(line.removeprefix("author "))
        elif line.startswith("committer "):
            committer = parse_identity(line.removeprefix("committer "))
        elif line.startswith("gpgsig "):
            raise GitError(f"Коммит {sha[:12]} подписан; переписывание подписи не поддержано")

    if not tree or author is None or committer is None:
        raise GitError(f"Коммит {sha[:12]} имеет неожиданный формат")
    return CommitData(
        tree=tree,
        parents=tuple(parents),
        author=author,
        committer=committer,
        message=message,
    )


def identity_env(commit: CommitData) -> dict[str, str]:
    """Формирует окружение для сохранения авторства и дат коммита.

    Args:
        commit: Данные коммита.

    Returns:
        Словарь переменных окружения для ``git commit-tree``.
    """
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": commit.author.name,
            "GIT_AUTHOR_EMAIL": commit.author.email,
            "GIT_AUTHOR_DATE": commit.author.date,
            "GIT_COMMITTER_NAME": commit.committer.name,
            "GIT_COMMITTER_EMAIL": commit.committer.email,
            "GIT_COMMITTER_DATE": commit.committer.date,
        }
    )
    return env


def ensure_clean_tree() -> None:
    """Проверяет отсутствие незакоммиченных изменений отслеживаемых файлов.

    Raises:
        GitError: Если рабочее дерево или индекс содержат изменения.
    """
    status = run_git("status", "--porcelain", "--untracked-files=no")
    if status.strip():
        raise GitError("Есть незакоммиченные изменения. Закоммитьте или отложите их перед переписыванием.")


def current_branch() -> str:
    """Возвращает имя текущей ветки.

    Returns:
        Имя текущей локальной ветки.

    Raises:
        GitError: Если HEAD отделён от ветки.
    """
    branch = run_git("symbolic-ref", "--quiet", "--short", "HEAD", check=False).strip()
    if not branch:
        raise GitError("HEAD не указывает на ветку; переключитесь на локальную ветку.")
    return branch


def resolve_commits(rev_range: str) -> list[str]:
    """Возвращает коммиты диапазона от старых к новым.

    Args:
        rev_range: Диапазон git, например ``@{upstream}..HEAD``.

    Returns:
        Список хешей коммитов.

    Raises:
        GitError: Если диапазон некорректен.
    """
    try:
        output = run_git("rev-list", "--reverse", "--topo-order", rev_range)
    except GitError as exc:
        raise GitError(f"Не удалось разобрать диапазон {rev_range!r}: {exc}") from exc
    return output.split()


def subject(sha: str) -> str:
    """Возвращает первую строку сообщения коммита.

    Args:
        sha: Полный хеш коммита.

    Returns:
        Первая строка сообщения коммита.
    """
    return run_git("show", "-s", "--format=%s", sha).strip()


def backup_ref(branch: str) -> str:
    """Создаёт уникальное имя резервной ссылки.

    Args:
        branch: Имя текущей ветки.

    Returns:
        Имя резервной ссылки git.
    """
    timestamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
    safe_branch = re.sub(r"[^A-Za-z0-9._-]+", "-", branch)
    return f"refs/backup/codebuff-clean/{timestamp}-{safe_branch}"


def rewrite_commits(commits: list[str], branch: str) -> str:
    """Пересоздаёт коммиты с очищенными сообщениями.

    Args:
        commits: Хеши коммитов от старых к новым.
        branch: Имя текущей ветки.

    Returns:
        Хеш новой вершины ветки.

    Raises:
        GitError: Если переписывание не удалось.
    """
    old_head = commits[-1]
    ref = backup_ref(branch)
    run_git("update-ref", ref, old_head)
    print(f"Резервная ссылка: {ref}")

    mapping: dict[str, str] = {}
    changed_any = False
    for sha in commits:
        commit = read_commit(sha)
        if len(commit.parents) > 1:
            raise GitError(f"Коммит {sha[:12]} является merge-коммитом; скрипт не поддерживает merge.")
        message, changed = strip_footer(commit.message)
        if not changed and not changed_any:
            mapping[sha] = sha
            continue

        changed_any = True
        new_parents = tuple(mapping.get(parent, parent) for parent in commit.parents)
        args = ["commit-tree", commit.tree]
        for parent in new_parents:
            args.extend(["-p", parent])
        new_sha = run_git(*args, input_text=message, env=identity_env(commit)).strip()
        mapping[sha] = new_sha
        print(f"  {sha[:12]} -> {new_sha[:12]}  {subject(sha)}")

    new_head = mapping[old_head]
    if new_head == old_head:
        raise GitError("Сообщения не изменились")
    run_git(
        "update-ref",
        "-m",
        "remove Codebuff attribution",
        f"refs/heads/{branch}",
        new_head,
        old_head,
    )
    return new_head


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки.

    Returns:
        Разобранные аргументы командной строки.
    """
    parser = argparse.ArgumentParser(description="Удаляет подпись Codebuff из сообщений незапушенных коммитов.")
    parser.add_argument(
        "--range",
        default=DEFAULT_RANGE,
        help=f"Диапазон коммитов (по умолчанию: {DEFAULT_RANGE}).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Переписать историю. Без флага выполняется только предпросмотр.",
    )
    return parser.parse_args()


def fail(message: str) -> NoReturn:
    """Печатает ошибку и завершает скрипт.

    Args:
        message: Текст ошибки.

    Returns:
        Функция не возвращает значение.
    """
    print(f"Ошибка: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    """Точка входа скрипта."""
    args = parse_args()
    try:
        ensure_clean_tree()
        branch = current_branch()
        commits = resolve_commits(args.range)
        if not commits:
            print(f"В диапазоне {args.range} нет коммитов.")
            return

        affected = [sha for sha in commits if strip_footer(read_commit(sha).message)[1]]
        if not affected:
            print("Подпись Codebuff в сообщениях не найдена.")
            return

        print(f"Ветка: {branch}")
        print(f"Диапазон: {args.range}")
        print(f"Коммитов с подписью: {len(affected)} из {len(commits)}")
        for sha in affected:
            print(f"  {sha[:12]}  {subject(sha)}")

        if not args.apply:
            print("\nПредпросмотр. Для переписывания истории запустите с --apply.")
            return

        new_head = rewrite_commits(commits, branch)
        print(f"\nГотово: {new_head[:12]}")
        print("Для отправки используйте: git push --force-with-lease")
    except GitError as exc:
        fail(str(exc))


if __name__ == "__main__":
    main()
