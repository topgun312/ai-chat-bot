from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import tiktoken

from chatbot.config import ChatConfig


class Message(TypedDict):
    """
    Одно сообщение диалога в формате, ожидаемом API провайдера.

    Attributes:
        role: автор сообщения — `"user"` или `"assistant"`.
        content: текст сообщения.
    """

    role: str  # "user" | "assistant"
    content: str


# Аппроксимация числа токенов в строке через tiktoken.
# cl100k_base — кодировщик Claude / GPT-4; для других моделей может отличаться.

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Быстрый подсчёт токенов без вызова API."""
    return len(_enc.encode(text))


def count_message_tokens(msg: Message) -> int:
    """Токены одного сообщения с оверхедом роли (~4 токена на role-markers)."""
    return count_tokens(msg["content"]) + 4


@dataclass
class ConversationHistory:
    """
    Хранит историю диалога и управляет размером контекста.

    Sliding window: при превышении context_limit из начала удаляются
    целые ходы (пара user+assistant), пока история не влезет.
    Первый user-вопрос никогда не удаляется — он может быть нужен для контекста.
    """

    config: ChatConfig
    _messages: list[Message] = field(default_factory=list)

    # ── Добавление сообщений ────────────────────────────────────────

    def add_user(self, text: str) -> None:
        """Добавляет реплику пользователя и при необходимости обрезает историю."""
        self._messages.append({"role": "user", "content": text})
        self._prune_if_needed()

    def add_assistant(self, text: str) -> None:
        """Добавляет ответ ассистента в конец истории (без обрезки)."""
        self._messages.append({"role": "assistant", "content": text})

    # ── Чтение ─────────────────────────────────────────────────────

    @property
    def messages(self) -> list[Message]:
        """Список сообщений для передачи в API."""
        return list(self._messages)

    @property
    def total_tokens(self) -> int:
        """Суммарные токены всей истории."""
        return sum(count_message_tokens(m) for m in self._messages)

    @property
    def turn_count(self) -> int:
        """Число полных ходов (user + assistant)."""
        return sum(1 for m in self._messages if m["role"] == "user")

    # ── Управление размером ─────────────────────────────────────────

    def _prune_if_needed(self) -> int:
        """
        Удаляет самые ранние ходы пока история не влезет в context_limit.
        Возвращает число удалённых ходов.
        """
        pruned = 0
        while self.total_tokens > self.config.context_limit:
            removed = self._removed_oldest_turn()
            if not removed:
                break  # нечего удалять — одно сообщение превышает лимит
            pruned += 1
        return pruned

    def _removed_oldest_turn(self) -> bool:
        """
        Удаляет одну пару user+assistant с начала истории.
        Если история пуста или содержит только одно сообщение — возвращает False.
        """
        if len(self._messages) < 2:
            return False
        # Находим первую пару user → assistant
        for i, msg in enumerate(self._messages):
            if msg["role"] == "user" and i + 1 < len(self._messages):
                if self._messages[i + 1]["role"] == "assistant":
                    del self._messages[i : i + 2]
                    return True

        # Если пары нет (одиноко стоящий user) — удаляем его
        if self._messages[0]["role"] == "user":
            del self._messages[0]
            return True
        return False

    # ── Утилиты ────────────────────────────────────────────────────

    def clear(self) -> None:
        """Полная очистка истории."""
        self._messages.clear()

    def stats(self) -> str:
        """Строка со статистикой для отображения в REPL."""
        pct = self.total_tokens / self.config.context_limit * 100
        return (
            f"💬 {self.turn_count} ходов · "
            f"🔤 {self.total_tokens:,} / {self.config.context_limit:,} токенов"
            f"({pct:.0f}%)"
        )
