import pytest

from rag_system.domain.conversation_manager import Message, ToolCallMessage, ToolResponseMessage
from rag_system.infrastructure.conversation_manager import InMemoryConversationManager


class TestInMemoryConversationManager:
    @pytest.fixture
    def manager(self):
        return InMemoryConversationManager()

    def test_initial_state(self, manager: InMemoryConversationManager):
        assert manager.gradio_messages == []
        assert manager.get_conversation_history() == []
        assert manager.get_conversation_history() is not manager.gradio_messages

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
            Message(role="assistant", content="The capital of France is Paris."),
        ]
        manager.update_conversation(messages)
        history = manager.get_conversation_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "What is the capital of France?"
        assert history[1].role == "assistant"
        assert history[1].content == "The capital of France is Paris."

    def test_add_tool_call_message(self, manager: InMemoryConversationManager):
        """Test adding tool call messages to internal LLM history."""
        tool_call = ToolCallMessage(
            role="assistant",
            content="",
            tool_calls=[{
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{}"}
            }]
        )
        manager.add_tool_call_message(tool_call)
        
        internal_history = manager.get_internal_llm_history()
        assert len(internal_history) == 1
        assert internal_history[0] == tool_call

    def test_add_tool_response_message(self, manager: InMemoryConversationManager):
        """Test adding tool response messages to internal LLM history."""
        tool_response = ToolResponseMessage(
            role="tool",
            content="Tool response content",
            tool_call_id="call_123",
            name="test_tool"
        )
        manager.add_tool_response_message(tool_response)
        
        internal_history = manager.get_internal_llm_history()
        assert len(internal_history) == 1
        assert internal_history[0] == tool_response

    def test_get_internal_llm_history_returns_copy(self, manager: InMemoryConversationManager):
        """Test that get_internal_llm_history returns a copy, not the original list."""
        tool_call = ToolCallMessage(
            role="assistant",
            content="",
            tool_calls=[{
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{}"}
            }]
        )
        manager.add_tool_call_message(tool_call)
        
        internal_history = manager.get_internal_llm_history()
        assert internal_history is not manager.internal_llm_history

    def test_clear_internal_llm_history(self, manager: InMemoryConversationManager):
        """Test clearing internal LLM history."""
        tool_call = ToolCallMessage(
            role="assistant",
            content="",
            tool_calls=[{
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{}"}
            }]
        )
        manager.add_tool_call_message(tool_call)
        assert len(manager.get_internal_llm_history()) == 1
        
        manager.clear_internal_llm_history()
        assert len(manager.get_internal_llm_history()) == 0
