import json
from datetime import datetime
from pathlib import Path

from chatbot.history import ConversationHistory


def export_history(history: ConversationHistory, fmt: str = "md") -> Path:
    """
    Сохраняет историю диалога в файл `chat_<timestamp>.<fmt>`.

    Поддерживаемые форматы: `md` (читаемый Markdown) и `json`
    (структурированные данные со служебной информацией).

    Args:
        history: история диалога для экспорта.
        fmt: формат файла — `"md"` или `"json"`.

    Returns:
        Путь к созданному файлу.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(f"chat_{ts}.{fmt}")

    if fmt == "json":
        data = {
            "exported_at": ts,
            "turns": history.turn_count,
            "messages": history.messages,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    elif fmt == "md":
        lines = [f"# Диалог {ts}\n"]
        for msg in history.messages:
            role = "**Вы**" if msg["role"] == "user" else "**Бот**"
            lines.append(f"{role}: {msg['content']}\n\n---\n")
        path.write_text("".join(lines), encoding="utf-8")

    return path
