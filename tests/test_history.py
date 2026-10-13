import pytest

from chatbot.config import ChatConfig
from chatbot.history import ConversationHistory, count_tokens


@pytest.fixture
def config() -> ChatConfig:
    return ChatConfig(api_key="test-key", context_limit=500)


@pytest.fixture
def history(config) -> ConversationHistory:
    return ConversationHistory(config)


class TestTokenCounting:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_short_text(self):
        assert count_tokens("hello") > 0

    def test_long_text_more_tokens(self):
        short = count_tokens("Hi")
        long = count_tokens("Hello, how are you doing today?")
        assert long > short


class TestConversationHistory:
    def test_add_user_message(self, history):
        history.add_user("привет")
        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "user"
        assert history.messages[0]["content"] == "привет"

    def test_add_assistant_message(self, history):
        history.add_user("Привет")
        history.add_assistant("Здравствуйте")
        assert len(history.messages) == 2
        assert history.messages[1]["role"] == "assistant"

    def test_turn_count(self, history):
        history.add_user("Вопрос 1")
        history.add_assistant("Ответ 1")
        history.add_user("Вопрос 2")
        history.add_assistant("Ответ 2")
        assert history.turn_count == 2

    def test_clear(self, history):
        history.add_user("Вопрос 1")
        history.clear()
        assert len(history.messages) == 0
        assert history.turn_count == 0

    def test_sliding_window_prunes_oldest(self, config):
        """При превышении лимита удаляются самые ранние ходы."""
        h = ConversationHistory(config)  # limit=500 токенов
        # Добавляем много коротких сообщений — каждое ~10 токенов
        for i in range(60):
            h.add_user(f"Вопрос номер {i}")
            h.add_assistant(f"Ответ номер {i}")
        # История не может превышать лимит
        assert h.total_tokens <= config.context_limit * 1.1  # 10% запас

    def test_pruning_preserves_latest(self, config):
        """После обрезки последние сообщения должны остаться."""
        h = ConversationHistory(config)  # limit=500
        for i in range(60):
            h.add_user(f"Вопрос  {i}")
            h.add_assistant(f"Ответ  {i}")
        # Последнее сообщение должно сохраниться
        last_messages = h.messages[-1]
        assert "59" in last_messages["content"]
