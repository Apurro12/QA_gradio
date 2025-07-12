from unittest.mock import patch

import pytest

from rag_system.domain.agent import BaseAgent
from rag_system.domain.conversation_manager import Message
from rag_system.infrastructure.conversation_manager import InMemoryConversationManager
from rag_system.use_cases.llm_client import OfflineLLMClient

from rag_system.use_cases.agent import Agent
from langchain_openai import ChatOpenAI
from unittest.mock import MagicMock, patch, call

## TO DO
## ADD THE CASE WHERE THE LOADED GIVES NO DOCUMENTS
@pytest.fixture
def offline_agent():
    llm = OfflineLLMClient(ChatOpenAI())
    return Agent(llm, None)


@pytest.fixture
def offline_agent_with_conversation_manager():
    llm = OfflineLLMClient(ChatOpenAI())
    in_memory_conversation_manager = InMemoryConversationManager()
    return Agent(llm, in_memory_conversation_manager)


def test_offline_agent_find_response(offline_agent: BaseAgent):
    empty_history: list[Message] = []
    message = "How can I get a refund?"
    response = offline_agent.chat(message, empty_history)
    assert isinstance(response, str)

    expected_offline_response = f"Last message is: '{message}'. Available tools: '[]'"
    assert expected_offline_response == response



def test_offline_agent_not_find_response(offline_agent: BaseAgent):
    empty_history: list[Message] = []
    message = "How can I get a refund?"
    response = offline_agent.chat(message, empty_history)
    assert isinstance(response, str)

    expected_offline_response = f"Last message is: '{message}'. Available tools: '[]'"
    assert expected_offline_response == response


def test_chat_with_no_conversation_manager(offline_agent: BaseAgent):
    empty_history: list[Message] = []
    return_value = "some response"
    with patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
        message = "Test question?"
        result = offline_agent.chat(message, empty_history)
        mock_chat_openai.assert_not_called()


def test_chat_with_conversation_manager_no_history_no_context(
    offline_agent_with_conversation_manager: BaseAgent,
):
    empty_history: list[Message] = []
    message = "First message"
    result = offline_agent_with_conversation_manager.chat(message, empty_history)
    expected_offline_response = f"Last message is: '{message}'. Available tools: '[]'"
    assert result == expected_offline_response


def test_agent_chat(offline_agent: BaseAgent):
    """Test that the tracer provider is initialized correctly."""
    message = "hello"
    response = offline_agent.chat(message, [])

    expected_respose = OfflineLLMClient(ChatOpenAI()).invoke(message)
    assert response == expected_respose

def test_agent_chat_with_tracing():
    """Test that tracing spans and attributes are set correctly."""
    from openinference.semconv.trace import SpanAttributes
    # Create a mock tracer provider and tracer
    mock_tracer_provider = MagicMock()
    mock_tracer_provider.bool = True  # Simulate a valid tracer provider

    mock_tracer = MagicMock()
    mock_span = MagicMock()
    
    

    llm = OfflineLLMClient(ChatOpenAI())
    agent = Agent(llm, None, tracer_provider=mock_tracer_provider)
    agent.tracer = mock_tracer
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    
    # Test with session_id
    message = "Test message"
    session_id = "test-session-123"
    response = agent.chat(message, [], session_id=session_id)
    
    # Verify span was started
    mock_tracer.start_as_current_span.assert_called_once_with("agent_chat")

    # Verify attributes were set

# Assert individual calls
    mock_span.set_attribute.assert_has_calls([
            call(SpanAttributes.SESSION_ID, session_id), 
            call(SpanAttributes.INPUT_VALUE, message), 
            call(SpanAttributes.OUTPUT_VALUE, response)
        ])


def test_agent_chat_with_tracing_and_conversation_manager():
    """Test that tracing works with conversation manager."""
    from openinference.semconv.trace import SpanAttributes
    
    # Create mocks
    mock_tracer_provider = MagicMock()
    mock_tracer_provider.bool = True
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_conversation_manager = MagicMock()
    
    # Setup agent with conversation manager
    llm = OfflineLLMClient(ChatOpenAI())
    agent = Agent(llm, mock_conversation_manager, tracer_provider=mock_tracer_provider)
    agent.tracer = mock_tracer
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    
    # Test with conversation manager
    message = "Test message"
    session_id = "test-session-123"
    response = agent.chat(message, [], session_id=session_id)
    
    # Verify conversation manager was called
    mock_conversation_manager.update_conversation.assert_called_once()
    
    # Verify the conversation update includes both user and assistant messages
    call_args = mock_conversation_manager.update_conversation.call_args[0][0]
    assert len(call_args) == 2
    assert call_args[0].role == "user"
    assert call_args[0].content == message
    assert call_args[1].role == "assistant"
    assert call_args[1].content == response
    