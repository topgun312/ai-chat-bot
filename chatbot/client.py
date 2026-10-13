from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionChunk

from chatbot.config import ChatConfig
from chatbot.history import Message, count_message_tokens, count_tokens

OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass
class StreamStats:
    """Статистика завершённого запроса."""

    input_tokens: int
    output_tokens: int
    stop_reason: str

    @property
    def total_tokens(self) -> int:
        """Суммарное число токенов запроса (вход + выход)."""
        return self.input_tokens + self.output_tokens


class ChatClient:
    """
    Async streaming клиент с тремя провайдерами, выбираемыми по config.provider:

      - ollama    — локальные модели; Ollama принимает OpenAI-формат,
                    поэтому используется AsyncOpenAI с другим base_url
      - openai    — официальный API OpenAI (AsyncOpenAI, стандартный base_url)
      - anthropic — официальный API Claude (AsyncAnthropic)
    """

    def __init__(self, config: ChatConfig) -> None:
        """Создаёт SDK-клиент, соответствующий `config.provider`."""
        self.config = config
        self._client: AsyncOpenAI | AsyncAnthropic = self._build_client(config)
        self._last_stats: StreamStats | None = None

    @staticmethod
    def _build_client(config: ChatConfig) -> AsyncOpenAI | AsyncAnthropic:
        """
        Фабрика SDK-клиентов по провайдеру.

        Args:
            config: конфигурация с провайдером и ключами.

        Returns:
            `AsyncAnthropic` для anthropic, иначе `AsyncOpenAI`
            (для ollama — с base_url локального сервера).

        Raises:
            ValueError: неизвестное имя провайдера.
        """
        match config.provider:
            case "anthropic":
                return AsyncAnthropic(api_key=config.api_key)
            case "openai":
                return AsyncOpenAI(api_key=config.openai_api_key)
            case "ollama":
                base_url = config.ollama_base_url.rstrip("/")
                return AsyncOpenAI(api_key="ollama", base_url=f"{base_url}/v1")
            case other:
                raise ValueError(f"Неизвестный провайдер: {other!r}")

    @property
    def last_stats(self) -> StreamStats | None:
        """Статистика последнего завершённого запроса (или None до первого)."""
        return self._last_stats

    # ── Публичный API ────────────────────────────────────────────────

    async def stream(
        self,
        messages: list[Message],
        system_prompt: str,
    ) -> AsyncGenerator[str, None]:
        """
        Стримит ответ модели, выдавая текстовые чанки по мере генерации.
        Сохраняет статистику в self._last_stats по завершении.

        Использование:
            async for chunk in client.stream(messages, system):
                print(chunk, end="", flush=True)
        """
        if self.config.provider == "anthropic":
            async for chunk in self._stream_anthropic(messages, system_prompt):
                yield chunk
        else:
            async for chunk in self._stream_openai(messages, system_prompt):
                yield chunk

    async def send(
        self,
        messages: list[Message],
        system_prompt: str,
        retries: int = 2,
    ) -> str:
        """
        Неблокирующий вызов без стриминга — для тестов и пакетной обработки.
        Возвращает полный текст ответа.
        """
        if self.config.provider == "anthropic":
            return await self._send_anthropic(messages, system_prompt, retries)
        return await self._send_openai(messages, system_prompt, retries)

    # ── OpenAI-совместимый путь (openai + ollama) ────────────────────

    async def _stream_openai(
        self,
        messages: list[Message],
        system_prompt: str,
    ) -> AsyncGenerator[str, None]:
        """Стрим OpenAI-совместимого API с обработкой ошибок."""
        try:
            async for chunk in self._openai_stream_raw(messages, system_prompt):
                yield chunk
        except openai.RateLimitError:
            yield "\n[⚠ Rate limit. Попробуйте через несколько секунд.]"
        except openai.AuthenticationError:
            yield "\n[⚠ Неверный OPENAI_API_KEY.]"
        except openai.APIStatusError as e:
            yield f"\n[⚠ API ошибка {e.status_code}: {e.message}]"
        except asyncio.CancelledError:
            return  # пользователь прервал (Ctrl+C)

    async def _openai_stream_raw(
        self,
        messages: list[Message],
        system_prompt: str,
    ) -> AsyncGenerator[str, None]:
        """Собственный стрим: чанки текста + сбор статистики по завершении."""
        full_text: list[str] = []
        stop_reason = "unknown"
        usage: CompletionUsage | None = None

        response = await self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                *messages,
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in response:
            text, finished = self._openai_chunk_parts(chunk)
            if text:
                full_text.append(text)
                yield text
            if finished:
                stop_reason = finished
            if chunk.usage:  # usage приходит в последнем чанке
                usage = chunk.usage

        self._last_stats = self._build_stats(
            usage, messages, system_prompt, full_text, stop_reason
        )

    @staticmethod
    def _openai_chunk_parts(chunk: ChatCompletionChunk) -> tuple[str, str]:
        """Разбирает чанк: (текст, finish_reason или '', если не финальный)."""
        if not chunk.choices:
            return "", ""
        delta = chunk.choices[0].delta
        return (delta.content or ""), (chunk.choices[0].finish_reason or "")

    @staticmethod
    def _build_stats(
        usage: CompletionUsage | None,
        messages: list[Message],
        system_prompt: str,
        full_text: list[str],
        stop_reason: str,
    ) -> StreamStats:
        """
        Собирает статистику запроса: из usage API или локальной оценки.

        Args:
            usage: токены от API (None, если провайдер не вернул usage).
            messages: история, отправленная в запрос.
            system_prompt: системный промпт запроса.
            full_text: собранные чанки ответа.
            stop_reason: причина завершения генерации.

        Returns:
            Статистика входных/выходных токенов.
        """
        if usage:
            return StreamStats(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                stop_reason=stop_reason,
            )
        # Fallback: некоторые версии Ollama не отдают usage в стриме —
        # оцениваем токены локально через tiktoken
        return StreamStats(
            input_tokens=(
                sum(count_message_tokens(m) for m in messages)
                + count_tokens(system_prompt)
            ),
            output_tokens=count_tokens("".join(full_text)),
            stop_reason=stop_reason,
        )

    async def _send_openai(
        self,
        messages: list[Message],
        system_prompt: str,
        retries: int = 2,
    ) -> str:
        """
        Нестриминговый запрос к OpenAI-совместимому API с retry.

        Args:
            messages: история диалога.
            system_prompt: системный промпт.
            retries: число повторов при rate limit (экспоненциальная задержка).

        Returns:
            Полный текст ответа модели.
        """
        payload_messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]
        for attempt in range(retries + 1):
            try:
                response = await self._client.chat.completions.create(  # type: ignore[union-attr]
                    model=self.config.model,
                    messages=payload_messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                choice = response.choices[0]
                usage = response.usage
                self._last_stats = StreamStats(
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    stop_reason=choice.finish_reason or "unknown",
                )
                return choice.message.content or ""

            except openai.RateLimitError:
                if attempt < retries:
                    await asyncio.sleep(2**attempt)
                else:
                    raise

        return ""  # unreachable

    # ── Anthropic-путь ────────────────────────────────────────────────

    async def _stream_anthropic(
        self,
        messages: list[Message],
        system_prompt: str,
    ) -> AsyncGenerator[str, None]:
        """
        Стрим ответа Claude: чанки текста + статистика по завершении.

        Ошибки API превращаются в текстовые сообщения для пользователя.
        """
        full_text: list[str] = []
        try:
            async with self._client.messages.stream(  # type: ignore[union-attr]
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=messages,  # type: ignore[arg-type]
            ) as stream:
                async for text_chunk in stream.text_stream:
                    full_text.append(text_chunk)
                    yield text_chunk

                final = await stream.get_final_message()
                self._last_stats = StreamStats(
                    input_tokens=final.usage.input_tokens,
                    output_tokens=final.usage.output_tokens,
                    stop_reason=final.stop_reason or "unknown",
                )

            if self._last_stats is None:
                self._last_stats = StreamStats(
                    input_tokens=(
                        sum(count_message_tokens(m) for m in messages)
                        + count_tokens(system_prompt)
                    ),
                    output_tokens=count_tokens("".join(full_text)),
                    stop_reason="unknown",
                )

        except anthropic.RateLimitError:
            yield "\n[⚠ Rate limit. Попробуйте через несколько секунд.]"
        except anthropic.AuthenticationError:
            yield "\n[⚠ Неверный ANTHROPIC_API_KEY.]"
        except anthropic.APIStatusError as e:
            yield f"\n[⚠ API ошибка {e.status_code}: {e.message}]"
        except asyncio.CancelledError:
            return  # пользователь прервал (Ctrl+C)

    async def _send_anthropic(
        self,
        messages: list[Message],
        system_prompt: str,
        retries: int = 2,
    ) -> str:
        """
        Нестриминговый запрос к Claude API с retry при rate limit.

        Args:
            messages: история диалога.
            system_prompt: системный промпт.
            retries: число повторов при rate limit (экспоненциальная задержка).

        Returns:
            Склеенный текст всех text-блоков ответа.
        """
        for attempt in range(retries + 1):
            try:
                response = await self._client.messages.create(  # type: ignore[union-attr]
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=system_prompt,
                    messages=messages,  # type: ignore[arg-type]
                )
                self._last_stats = StreamStats(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    stop_reason=response.stop_reason or "unknown",
                )
                return "".join(
                    block.text for block in response.content if block.type == "text"
                )

            except anthropic.RateLimitError:
                if attempt < retries:
                    await asyncio.sleep(2**attempt)
                else:
                    raise

        return ""  # unreachable


async def list_ollama_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """Получить список доступных моделей Ollama (через /v1/models)."""
    client = AsyncOpenAI(api_key="ollama", base_url=f"{base_url}/v1")
    return [m.id async for m in await client.models.list()]
