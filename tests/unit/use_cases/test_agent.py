from unittest.mock import patch

import pytest

from rag_system.domain.agent import BaseAgent
from rag_system.domain.conversation_manager import Message
from rag_system.infrastructure.conversation_manager import InMemoryConversationManager
from rag_system.use_cases.llm_client import OfflineLLMClient

from rag_system.use_cases.agent import Agent
from langchain_openai import ChatOpenAI

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
