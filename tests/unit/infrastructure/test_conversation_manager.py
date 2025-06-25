import pytest
from rag_system.domain.conversation_manager import Message
from rag_system.infrastructure.conversation_manager import InMemoryConversationManager

class TestInMemoryConversationManager:
    @pytest.fixture
    def manager(self):
        return InMemoryConversationManager()

    def test_initial_state(self, manager: InMemoryConversationManager):
        assert manager.messages == []
        assert manager.get_conversation_history() == []
        assert manager.get_conversation_history() is not manager.messages

    def test_add_and_get_conversation_history(self, manager: InMemoryConversationManager):
        manager.add_message(Message(role="user", content="Hello!"))
        manager.add_message(Message(role="assistant", content="Hi there!"))
        history = manager.get_conversation_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Hello!"
        assert history[0].timestamp is not None
        assert isinstance(history[0].timestamp, float)

        assert history[1].role == "assistant"
        assert history[1].content == "Hi there!"
        assert history[1].timestamp is not None
        assert isinstance(history[1].timestamp, float)

    def test_clear_conversation(self, manager: InMemoryConversationManager):
        manager.add_message(Message(role="user", content="Hello!"))
        manager.add_message(Message(role="assistant", content="Hi!"))
        manager.clear_conversation()
        assert manager.get_conversation_history() == []

    def test_update_conversation(self, manager: InMemoryConversationManager):

        manager.add_message(Message(role="user", content="Hello!"))
        assert len(manager.get_conversation_history()) == 1
        assert manager.get_conversation_history()[0].role == "user"
        assert manager.get_conversation_history()[0].content == "Hello!"

        messages = [
            Message(role="user", content="What is the capital of France?"),
            Message(role="assistant", content="The capital of France is Paris.")
        ]
        manager.update_conversation(messages)
        history = manager.get_conversation_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "What is the capital of France?"
        assert history[1].role == "assistant"
        assert history[1].content == "The capital of France is Paris."

