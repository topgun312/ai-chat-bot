from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatConfig(BaseSettings):
    """
    Настройки бота: чтение из `.env` (pydantic-settings) и валидация.

    Поля-алиасы (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OLLAMA_BASE_URL`)
    читаются из окружения/`.env`; имя поля можно передавать и напрямую.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,  # принимать и имя поля, и его env-алиас
    )

    # ── Провайдер ────────────────────────────────────────────────────
    # ollama    — локальные модели (по умолчанию, ничего настраивать не надо)
    # openai    — официальный API OpenAI (нужен OPENAI_API_KEY)
    # anthropic — официальный API Claude (нужен ANTHROPIC_API_KEY)
    provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    model: str = "llama3.2"
    api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )

    # ── Параметры генерации ─────────────────────────────────────────
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    # ── Управление контекстом ───────────────────────────────────────
    # Максимум токенов на историю: сигнал для sliding window
    # Оставляем запас: system prompt (~300) + ответ (~1024)
    context_limit: int = Field(default=6000, ge=500)
    prune_turns: int = Field(default=2, ge=1)

    # ── Персонаж бота ───────────────────────────────────────────────
    bot_name: str = "Эмилия"
    bot_role: str = "помощник AI-разработчика"

    @model_validator(mode="after")
    def check_provider_key(self) -> Self:
        """Для облачных провайдеров ключ обязателен, для ollama — нет."""
        if self.provider == "anthropic" and not self.api_key:
            raise ValueError(
                "provider='anthropic' требует ключ ANTHROPIC_API_KEY в .env"
            )
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("provider='openai' требует ключ OPENAI_API_KEY в .env")
        return self

    @model_validator(mode="after")
    def check_model_matches_provider(self) -> Self:
        """Предупреждение, если модель не похожа на модель выбранного провайдера."""
        prefixes = {
            "anthropic": ("claude",),
            "openai": ("gpt", "o1", "o3", "o4", "chatgpt"),
        }
        expected = prefixes.get(self.provider)
        if expected and not self.model.startswith(expected):
            import warnings

            warnings.warn(
                f"Модель {self.model!r} не похожа на модель провайдера "
                f"{self.provider!r} (ожидается префикс: {', '.join(expected)})"
            )
        return self

    # Предупреждение, если модель не из известного списка
    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Предупреждает, если имя модели не входит в список известных."""
        known = {
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "gpt-4o",
            "gpt-4o-mini",
            "llama3.2",
        }

        if v not in known:
            import warnings

            warnings.warn(f"Незнакомая модель: {v!r}. Проверьте правильность")
        return v
