# ai-history-chat-bot

Асинхронный консольный чат-бот на Python 3.12 с тремя переключаемыми провайдерами LLM — **Ollama** (локальные модели), **OpenAI** и **Anthropic Claude** — управлением контекстом (sliding window), защитой ввода от prompt injection и потоковым (streaming) выводом ответа.

## Возможности

- 💬 **REPL-чат в терминале** — асинхронный цикл диалога со стримингом ответа модели в реальном времени.
- 🧠 **История диалога** — контекст сохраняется между сообщениями; при превышении лимита токенов старые ходы (пара user + assistant) автоматически удаляются из начала истории.
- 🔢 **Подсчёт токенов** — быстрая оценка через `tiktoken` (`cl100k_base`) без вызова API, плюс точная статистика по каждому ответу от API.
- 🛡️ **Guard ввода** — сканирование пользовательского ввода на попытки prompt injection («ignore previous instructions», «покажи системный промпт», XML-инъекции и т.п.), санитизация HTML/XML-спецсимволов, ограничение длины ввода.
- 📝 **XML-структурированный системный промпт** — имя и роль бота настраиваются через конфиг.
- ⚙️ **Конфигурация через `.env`** — валидация настроек через `pydantic-settings`.
- 🔄 **Три провайдера LLM** — `ollama` / `openai` / `anthropic` переключаются одной строкой `PROVIDER` в `.env`.
- ⚡ **Режимы генерации** — `/mode chat|code|creative` переключает температуру на лету.
- 📤 **Экспорт диалога** — `/export` сохраняет историю в Markdown-файл.
- ⌨️ **Команды**: `/help`, `/stats`, `/clear`, `/mode`, `/export`, `/exit`.

## Структура проекта

```
├── .env.example        # пример конфигурации (копируется в .env)
├── .flake8             # правила линтера: max-complexity=10, строка ≤ 121
├── pyproject.toml      # зависимости + настройки black/isort
chatbot/
├── main.py           # Точка входа: async REPL, команды, обработка ввода
├── config.py         # ChatConfig (pydantic-settings) — чтение .env и валидация
├── history.py        # ConversationHistory — история диалога, sliding window, подсчёт токенов
├── client.py         # ChatClient — streaming-клиент для ollama/openai/anthropic, retry, статистика
├── export_history.py # export_history() — экспорт диалога в md/json
├── prompt.py         # build_system_prompt() — сборка системного промпта из конфига
└── safety.py         # InputGuard — защита от prompt injection и санитизация ввода
tests/
├── test_chatbot.py   # интеграционные тесты (нужен ANTHROPIC_API_KEY)
├── test_history.py
├── test_providers.py # выбор провайдера, ключи, маршрутизация stream/send
└── test_safety.py
```

## Требования

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) (менеджер пакетов и окружений)
- Для `PROVIDER=ollama` — запущенный [Ollama](https://ollama.com) с моделью (`ollama pull llama3.2`)
- Для `PROVIDER=openai` — `OPENAI_API_KEY`
- Для `PROVIDER=anthropic` — `ANTHROPIC_API_KEY`

Зависимости смотрите в `pyproject.toml` (локально ставятся через `uv sync`):
рантайм — `anthropic`, `openai`, `pydantic-settings`, `tiktoken`; инструменты — `pytest`, `flake8`, `black`, `isort`.

## Быстрый старт

```bash
# 1. Клонировать репозиторий и установить зависимости
git clone <repo-url>
cd ai-chat-bot
uv sync

# 2. (только для ollama) подтянуть модель
ollama pull llama3.2

# 3. Создать .env на основе примера
cp .env.example .env
# выбрать PROVIDER и вписать ключ (для облачных провайдеров)

# 4. Запустить бота
uv run python chatbot/main.py
# альтернатива: uv run python -m chatbot.main
```

> `.env` не коммитится в git (добавлен в `.gitignore`).

## Конфигурация

Настройки читаются из `.env` (см. `.env.example`):

| Переменная          | По умолчанию        | Описание                                        |
|---------------------|---------------------|-------------------------------------------------|
| `PROVIDER`          | `ollama`            | Провайдер: `ollama` \| `openai` \| `anthropic`  |
| `MODEL`             | `llama3.2`          | Модель (при незнакомом имени выдаётся warning)  |
| `ANTHROPIC_API_KEY` | — (нужен для `anthropic`) | API-ключ Anthropic                        |
| `OPENAI_API_KEY`    | — (нужен для `openai`)    | API-ключ OpenAI                           |
| `OLLAMA_BASE_URL`   | `http://localhost:11434`  | Адрес сервера Ollama                     |
| `TEMPERATURE`       | `0.8`               | Температура генерации (0.0–2.0)                 |
| `MAX_TOKENS`        | `1024`              | Максимум токенов ответа (64–8192)               |
| `CONTEXT_LIMIT`     | `6000`              | Лимит токенов истории для sliding window        |
| `PRUNE_TURNS`       | `2`                 | Объявлен, но пока не используется кодом          |
| `BOT_NAME`          | `Эмилия`            | Имя персонажа бота                              |
| `BOT_ROLE`          | `помощник AI-разработчика` | Роль персонажа                           |

### Провайдеры

`ChatClient` создаёт нужный SDK по `config.provider` (`chatbot/client.py`):

| `PROVIDER`  | SDK               | Что требуется                              |
|-------------|-------------------|--------------------------------------------|
| `ollama`    | `AsyncOpenAI` на `OLLAMA_BASE_URL/v1` | локальный Ollama, ключ не нужен |
| `openai`    | `AsyncOpenAI` (официальный endpoint)  | `OPENAI_API_KEY`                  |
| `anthropic` | `AsyncAnthropic`                      | `ANTHROPIC_API_KEY`               |

```bash
# локально (по умолчанию)
PROVIDER=ollama
MODEL=llama3.2

# облачный OpenAI
PROVIDER=openai
OPENAI_API_KEY=sk-...
MODEL=gpt-4o

# облачный Anthropic
PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
MODEL=claude-sonnet-4-6
```

Если у облачного провайдера нет ключа — бот не стартует, а сразу сообщает, какая переменная не задана. Если `MODEL` не похожа на модель выбранного провайдера (`claude*` для anthropic, `gpt*`/`o*` для openai) — выводится предупреждение.

## Команды чата

| Команда                     | Действие                                       |
|-----------------------------|------------------------------------------------|
| `/help`                     | Справка по командам                            |
| `/stats`                    | Токены истории и статистика последнего запроса |
| `/clear`                    | Полная очистка истории диалога                 |
| `/mode <chat\|code\|creative>` | Смена режима генерации: `t=0.9` / `0.1` / `1.3`; без аргумента — список режимов |
| `/export`                   | Сохранить диалог в `chat_<YYYYMMDD_HHMMSS>.md` |
| `/exit`, `/quit`            | Выход (также `Ctrl+C` / `Ctrl+D`)              |

Неизвестная команда не завершает бота — выводится подсказка ввести `/help`.

## Архитектурные детали

- **Провайдеры**: `ChatClient._build_client()` по `config.provider` создаёт `AsyncAnthropic` (anthropic) или `AsyncOpenAI` (openai и ollama — у ollama меняется только `base_url` на `OLLAMA_BASE_URL/v1`). Публичные `stream()` / `send()` маршрутизируют на нужную реализацию.
- **Sliding window**: перед добавлением каждого user-сообщения проверяется суммарное число токенов; если оно превышает `context_limit`, из начала удаляются целые ходы (user + assistant).
- **Стриминг**: `ChatClient.stream()` — async generator, выдающий текстовые чанки по мере генерации; после завершения сохраняет статистику (`input_tokens`, `output_tokens`, `stop_reason`). Если провайдер не вернул `usage`, токены оцениваются локально через `tiktoken`.
- **Retry**: метод `send()` повторяет запрос при `RateLimitError` с экспоненциальной задержкой (2ⁿ секунд).
- **Режимы**: `/mode` меняет `config.temperature`, и она передаётся в оба SDK при каждом запросе.
- **Уровни угрозы**: `SAFE` → пропуск, `SUSPECT` → предупреждение и отправка как есть, `BLOCKED` → отклонение сообщения.
- **Ошибки конфигурации**: отсутствие ключа у облачного провайдера — понятное сообщение и выход вместо traceback (`chatbot/main.py`); несовпадение `MODEL` и провайдера — предупреждение при старте.

## Тесты

```bash
uv run pytest tests/
```

- `tests/test_history.py` — подсчёт токенов, sliding window, `clear`/`stats`;
- `tests/test_safety.py` — guard: injection-паттерны, санитизация, лимиты;
- `tests/test_providers.py` — выбор SDK по `PROVIDER`, обязательность ключей, маршрутизация `stream`/`send`;
- `tests/test_chatbot.py` — интеграционные (маркер `integration`): пропускаются, если не задан `ANTHROPIC_API_KEY`.

Сейчас: 32 unit-теста проходят, 4 интеграционных пропускаются без ключа.

## Код-стайл

```bash
uv run flake8 chatbot/ tests/   # линтер: max-complexity=10, строка ≤ 121
uv run black .                  # форматирование
uv run isort .                  # сортировка импортов (профиль black)
```

## Roadmap / TODO

- [x] Поддержка провайдеров `ollama` / `openai` / `anthropic` (переключение через `PROVIDER`)
- [x] Юнит-тесты: история, guard ввода, провайдеры
- [ ] Команда `/export json` (JSON-экспорт уже реализован в `export_history()`, команда захардкожена на `md`)
- [ ] Либо реализовать `PRUNE_TURNS`, либо убрать поле из конфига
- [ ] Интеграционные тесты для `openai` и `ollama` (сейчас только `anthropic`)
