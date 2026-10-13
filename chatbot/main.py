# main.py
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from pydantic import ValidationError

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot.client import ChatClient
from chatbot.config import ChatConfig
from chatbot.export_history import export_history
from chatbot.history import ConversationHistory
from chatbot.prompt import build_system_prompt
from chatbot.safety import InputGuard, ThreatLevel

# ── Вспомогательные функции ──────────────────────────────────────────────────

HELP_TEXT = """
Команды:
  /help               — показать эту справку
  /stats              — токены и статистика диалога
  /clear              — очистить историю диалога
  /mode <chat|code|creative> — сменить режим генерации (temperature)
  /export             — сохранить диалог в Markdown-файл
  /exit, /quit        — выйти
"""

MODES = {
    "chat": {"temperature": 0.9, "desc": "непринуждённый разговор"},
    "code": {"temperature": 0.1, "desc": "точные технические ответы"},
    "creative": {"temperature": 1.3, "desc": "творческие задачи"},
}


def print_bot(name: str, text: str = "") -> None:
    """Печатает префикс бота (без перевода строки — для стриминга)."""
    if text:
        print(f"\n\033[1;35m{name}:\033[0m {text}")
    else:
        print(f"\n\033[1;35m{name}:\033[0m ", end="", flush=True)


def print_system(msg: str) -> None:
    """Системные сообщения — серым цветом."""
    print(f"\033[90m{msg}\033[0m")


def read_input(prompt: str = "\033[1;36mВы:\033[0m ") -> str:
    """
    Читает строку из stdin как байты и декодирует с errors="replace".
    Обход бага: input() из потока-исполнителя падает с UnicodeDecodeError
    на многобайтовых символах (кириллица) — читаем через buffer напрямую.
    """
    print(prompt, end="", flush=True)
    line = sys.stdin.buffer.readline()
    if not line:
        raise EOFError
    return line.decode("utf-8", errors="replace")


# ── Обработка команд ──────────────────────────────────────────────────────────


def handle_command(
    cmd: str,
    history: ConversationHistory,
    client: ChatClient,
    config: ChatConfig,
) -> bool:
    """
    Обрабатывает /команды.
    Возвращает True если нужно продолжить цикл (не выходить).
    """
    match cmd.strip().lower():
        case "/help":
            print(HELP_TEXT)
        case "/stats":
            print_system(history.stats())
            if client.last_stats:
                s = client.last_stats
                print_system(
                    f"Последний запрос: {s.input_tokens} вход + "
                    f"{s.output_tokens} выход = {s.total_tokens} токенов"
                )
        case "/clear":
            history.clear()
            print_system("История очищена.")
        case "/exit" | "/quit":
            print_system("Выход.")
            return False
        case cmd if cmd.startswith("/mode"):
            parts = cmd.split()
            if len(parts) < 2 or parts[1] not in MODES:
                available = ", ".join(MODES.keys())
                print_system(f"Доступные режимы: {available}")
            else:
                mode = MODES[parts[1]]
                config.temperature = mode["temperature"]
                print_system(
                    f"Режим: {parts[1]} ({mode['desc']}, t={mode['temperature']})"
                )
        case "/export":
            path = export_history(history, "md")
            print_system(f"Диалог сохранён: {path}")
        case _:
            print_system(f"Неизвестная команда: {cmd!r}. Введите /help.")
    return True


# ── Основной цикл ─────────────────────────────────────────────────────────────


async def chat_loop(
    config: ChatConfig,
    client: ChatClient,
    history: ConversationHistory,
    system_prompt: str,
) -> None:
    """Async REPL: читает ввод, вызывает модель, печатает ответ."""

    print_system(
        f"Чат-бот '{config.bot_name}' запущен. "
        f"Провайдер: {config.provider} · Модель: {config.model}"
    )
    print_system("Введите /help для списка команд.\n")

    while True:
        # ── Читаем ввод ───────────────────────────────────────────────
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, read_input
            )
        except (EOFError, KeyboardInterrupt):
            print()
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # ── Команды ───────────────────────────────────────────────────
        if user_input.startswith("/"):
            if not handle_command(user_input, history, client, config):
                break
            continue

        # ── Обычное сообщение ─────────────────────────────────────────
        await handle_message(user_input, config, client, history, system_prompt)


async def handle_message(
    user_input: str,
    config: ChatConfig,
    client: ChatClient,
    history: ConversationHistory,
    system_prompt: str,
) -> None:
    """Guard, добавление в историю, стриминг ответа и статистика."""
    level, clean_input = InputGuard.check(user_input)

    if level == ThreatLevel.BLOCKED:
        print_system(
            "⚠ Сообщение заблокировано: обнаружена попытка изменить инструкции."
        )
        return

    if level == ThreatLevel.SUSPECT:
        print_system(
            "⚠ Подозрительный ввод — отправляю как есть, соблюдаю осторожность."
        )

    # ── Добавляем в историю и обрезаем при необходимости ──────────────
    history.add_user(clean_input)

    # ── Стриминг ответа ───────────────────────────────────────────────
    response = await stream_reply(config, client, history, system_prompt)

    # ── Сохраняем ответ в историю ─────────────────────────────────────
    if response:
        history.add_assistant(response)

    # ── Показываем статистику ─────────────────────────────────────────
    print_system(history.stats())


async def stream_reply(
    config: ChatConfig,
    client: ChatClient,
    history: ConversationHistory,
    system_prompt: str,
) -> str:
    """Печатает ответ по чанкам и возвращает его целиком."""
    print_bot(config.bot_name)

    full_response: list[str] = []
    try:
        async for chunk in client.stream(history.messages, system_prompt):
            print(chunk, end="", flush=True)
            full_response.append(chunk)
    except KeyboardInterrupt:
        print("\n[прервано]")

    print()  # перевод строки после ответа
    return "".join(full_response)


# ── Точка входа ───────────────────────────────────────────────────────────────


async def main() -> None:
    """Точка входа: создаёт конфиг, клиент, историю и запускает REPL."""
    config = ChatConfig()  # читает .env автоматически
    client = ChatClient(config)
    history = ConversationHistory(config)
    system = build_system_prompt(config)

    await chat_loop(config, client, history, system)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except ValidationError as e:
        print(f"\033[91mОшибка конфигурации:\033[0m\n{e}", file=sys.stderr)
        sys.exit(1)
