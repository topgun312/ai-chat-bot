from __future__ import annotations

import html
import re
import unicodedata
from enum import Enum


class ThreatLevel(Enum):
    """Уровень угрозы пользовательского ввода."""

    SAFE = "safe"
    SUSPECT = "suspect"  # логируем, но пропускаем
    BLOCKED = "blocked"  # отклоняем, сообщаем пользователю


# Паттерны прямого prompt injection
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"игнорир(уй|уйте)\s+(все\s+)?предыдущие\s+инструкции",
    r"forget\s+(everything|all)",
    r"you\s+are\s+now\s+(a\s+)?(?!helpful)",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"покажи\s+(свой\s+)?системный\s+промпт",
    r"jailbreak",
    r"DAN\s+mode",
    r"<\s*/?system\s*>",  # попытка вставить XML system-тег
    r"<\s*/?instructions?\s*>",  # попытка перебить тег инструкций
]

_COMPLETED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS]


class InputGuard:
    """
    Минимальный guard для пользовательского ввода.
    """

    MAX_INPUT_LENGTH = 4000  # символов

    @classmethod
    def scan(cls, text: str) -> ThreatLevel:
        """Определяет уровень угрозы для входного текста."""
        # Нормализуем Unicode — убираем скрытые символы
        normalized = unicodedata.normalize("NFKC", text)

        if len(normalized) > cls.MAX_INPUT_LENGTH:
            return ThreatLevel.BLOCKED

        for pattern in _COMPLETED:
            if pattern.search(normalized):
                return ThreatLevel.BLOCKED

        # Подозрительные признаки без блокировки
        suspicious_signs = [
            "system_prompt" in normalized.lower(),
            "ignore instructions" in normalized.lower(),
            normalized.count("<") > 5,  # много тегов — возможная XML-инъекция
        ]

        if any(suspicious_signs):
            return ThreatLevel.SUSPECT

        return ThreatLevel.SAFE

    @classmethod
    def sanitize(cls, text: str) -> str:
        """
        Экранирует HTML/XML-спецсимволы в тексте.
        Предотвращает вырывание из XML-тегов промпта.
        """
        normalized = unicodedata.normalize("NFKC", text)
        return html.escape(normalized, quote=False)

    @classmethod
    def check(cls, text: str) -> tuple[ThreatLevel, str]:
        """
        Полная проверка: сканирование + санитизация.
        Возвращает (уровень, очищенный_текст).
        При BLOCKED очищенный текст пустой.
        """
        level = cls.scan(text)
        if level == ThreatLevel.BLOCKED:
            return level, ""
        return level, cls.sanitize(text)
