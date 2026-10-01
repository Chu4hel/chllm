"""Асинхронная шина сообщений между агентными сессиями (SessionMessageBus)."""

import asyncio

from .domain import AgentMessage


class SessionMessageBus:
    """Асинхронная шина сообщений (Mailbox) для межсессионного взаимодействия."""

    def __init__(self) -> None:
        """Инициализирует шину сообщений."""
        self._mailboxes: dict[str, asyncio.Queue[AgentMessage]] = {}

    def _get_queue(self, session_id: str) -> asyncio.Queue[AgentMessage]:
        """Возвращает или создает очередь почтового ящика для сессии."""
        if session_id not in self._mailboxes:
            self._mailboxes[session_id] = asyncio.Queue()
        return self._mailboxes[session_id]

    async def send_message(self, message: AgentMessage) -> None:
        """Отправляет сообщение в почтовый ящик получателя.

        Args:
            message: Экземпляр AgentMessage.
        """
        queue = self._get_queue(message.recipient_id)
        await queue.put(message)

    async def receive_message(self, session_id: str, timeout: float | None = None) -> AgentMessage | None:
        """Ожидает поступления входящего сообщения для указанной сессии.

        Args:
            session_id: Идентификатор сессии-получателя.
            timeout: Опциональный таймаут ожидания в секундах.

        Returns:
            Полученное сообщение AgentMessage или None, если истек таймаут.
        """
        queue = self._get_queue(session_id)
        if timeout is None:
            return await queue.get()

        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def read_inbox(self, session_id: str) -> list[AgentMessage]:
        """Неблокирующе извлекает все скопившиеся в ящике сообщения.

        Args:
            session_id: Идентификатор сессии.

        Returns:
            Список накопившихся сообщений.
        """
        queue = self._get_queue(session_id)
        messages: list[AgentMessage] = []
        while not queue.empty():
            try:
                messages.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    def has_messages(self, session_id: str) -> bool:
        """Проверяет, есть ли непрочитанные сообщения в ящике сессии.

        Args:
            session_id: Идентификатор сессии.

        Returns:
            True, если ящик не пуст, иначе False.
        """
        return not self._get_queue(session_id).empty()

    def clear_mailbox(self, session_id: str) -> None:
        """Очищает почтовый ящик сессии.

        Args:
            session_id: Идентификатор сессии.
        """
        if session_id in self._mailboxes:
            queue = self._mailboxes[session_id]
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
