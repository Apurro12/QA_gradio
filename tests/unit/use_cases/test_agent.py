from typing import List
import pytest
from rag_system.domain.agent import BaseAgent
from rag_system.domain.conversation_manager import Message
from rag_system.interfaces.strategies_classifier.exact_match_classifier import ExactMatchClassifier
from rag_system.use_cases.agent import Agent
from rag_system.use_cases.answer_generator import OfflineAnswerGenerator
from rag_system.infrastructure.document_loader import OfflineDocumentLoader
from rag_system.infrastructure.llm_client import OfflineLLMClient
from rag_system.infrastructure.conversation_manager import InMemoryConversationManager
from unittest.mock import patch

## TO DO
## ADD THE CASE WHERE THE LOADED GIVES NO DOCUMENTS

@pytest.fixture
def offline_agent():
    llm = OfflineLLMClient()
    offline_document_loader = OfflineDocumentLoader()
    question_classifier = ExactMatchClassifier(document_loader=offline_document_loader)
    answer_generator = OfflineAnswerGenerator(llm=llm)
    return Agent(question_classifier, answer_generator)

@pytest.fixture
def offline_agent_with_conversation_manager():
    llm = OfflineLLMClient()
    offline_document_loader = OfflineDocumentLoader()
    question_classifier = ExactMatchClassifier(document_loader=offline_document_loader)
    answer_generator = OfflineAnswerGenerator(llm=llm)
    in_memory_conversation_manager = InMemoryConversationManager()
    return Agent(question_classifier, answer_generator, in_memory_conversation_manager)



def test_offline_agent_find_response(offline_agent: BaseAgent):

    empty_history: List[Message] = []
    response = offline_agent.respond_user_question("How can I get a refund?", empty_history)
    assert isinstance(response, str)

    expected_offline_response = f"[OFFLINE ANSWER GENERATOR] [OFFLINE LLM CALL]"

    assert expected_offline_response == response

def test_offline_agent_not_find_response(offline_agent: BaseAgent):
    empty_history: List[Message] = []
    response = offline_agent.respond_user_question("What is the capital of France?", empty_history)
    assert isinstance(response, str)

    expected_offline_response = 'No relevant information found to answer your question.'
    assert expected_offline_response == response


def test_chat_with_no_conversation_manager(offline_agent: BaseAgent):
    empty_history: List[Message] = []
    return_value = 'some response'
    with patch.object(offline_agent, "respond_user_question", return_value=return_value) as mock_respond:
        message = "Test question?"
        result = offline_agent.chat(message, empty_history)
        mock_respond.assert_called_once_with(message, empty_history)
        assert result == return_value



def test_chat_with_conversation_manager(offline_agent_with_conversation_manager: BaseAgent):
    empty_history: List[Message] = []
    message = "First message"
    result = offline_agent_with_conversation_manager.chat(message, empty_history)
    assert result == "[OFFLINE ANSWER GENERATOR] [OFFLINE LLM CALL]"



def test_chat_with_conversation_manager(offline_agent_with_conversation_manager: Agent):
    empty_history: List[Message] = []
    first_message = "First message"
    # Patch the generate method to spy on its call
    with patch.object(
        offline_agent_with_conversation_manager.answer_generator, 
        "generate", 
        return_value="[OFFLINE ANSWER GENERATOR] [OFFLINE LLM CALL]"
    ) as mock_generate:
        result = offline_agent_with_conversation_manager.chat(first_message, empty_history)
        mock_generate.assert_called_once()
        # You can check the exact arguments passed
        called_args, _ = mock_generate.call_args
        assert called_args[0] == first_message
        assert called_args[1] == ""
        # Optionally, check that full_context contains expected substrings
        #assert "Relevant Information:" in called_args[1]
        assert result == "[OFFLINE ANSWER GENERATOR] [OFFLINE LLM CALL]"

def test_chat_with_conversation_manager_with_context(offline_agent_with_conversation_manager: Agent):
    empty_history: List[Message] = []
    first_message = "When will my package arrive?"
    # Patch the generate method to spy on its call
    with patch.object(
        offline_agent_with_conversation_manager.answer_generator, 
        "generate", 
        return_value="[OFFLINE ANSWER GENERATOR] [OFFLINE LLM CALL]"
    ) as mock_generate:
        result = offline_agent_with_conversation_manager.chat(first_message, empty_history)
        mock_generate.assert_called_once()
        # You can check the exact arguments passed
        called_args, _ = mock_generate.call_args
        assert called_args[0] == first_message
        assert called_args[1] == "Relevant Information:\nShipping takes 3-5 business days."
        # Optionally, check that full_context contains expected substrings
        #assert "Relevant Information:" in called_args[1]
        assert result == "[OFFLINE ANSWER GENERATOR] [OFFLINE LLM CALL]"


def test_chat_with_conversation_manager_with_history(offline_agent_with_conversation_manager: Agent):
    history: List[Message] = [
        Message(role="user", content="Message1"), 
        Message(role="assistant", content="Response1")
    ]
    user_message = "Message2"
    # Patch the generate method to spy on its call
    with patch.object(
        offline_agent_with_conversation_manager.answer_generator, 
        "generate", 
        return_value="[OFFLINE ANSWER GENERATOR] [OFFLINE LLM CALL]"
    ) as mock_generate:
        result = offline_agent_with_conversation_manager.chat(user_message, history)
        mock_generate.assert_called_once()
        # You can check the exact arguments passed
        called_args, _ = mock_generate.call_args
        assert called_args[0] == user_message
        # This should be moved to a funcion please
        assert called_args[1] == f"Conversation History:\n{history[0].role}: {history[0].content}\n{history[1].role}: {history[1].content}\n \n"
        assert result == "[OFFLINE ANSWER GENERATOR] [OFFLINE LLM CALL]"

        

def test_chat_with_conversation_manager_with_history_and_context(offline_agent_with_conversation_manager: Agent):
    history: List[Message] = [
        Message(role="user", content="Message1"), 
        Message(role="assistant", content="Response1")
    ]
    user_message = "When will my package arrive?"
    # Patch the generate method to spy on its call
    with patch.object(
        offline_agent_with_conversation_manager.answer_generator, 
        "generate", 
        return_value="[OFFLINE ANSWER GENERATOR] [OFFLINE LLM CALL]"
    ) as mock_generate:
        result = offline_agent_with_conversation_manager.chat(user_message, history)
        mock_generate.assert_called_once()
        # You can check the exact arguments passed
        called_args, _ = mock_generate.call_args
        assert called_args[0] == user_message
        # This should be moved to a funcion please
        assert called_args[1] == f"Conversation History:\n{history[0].role}: {history[0].content}\n{history[1].role}: {history[1].content}\n \nRelevant Information:\nShipping takes 3-5 business days."
        assert result == "[OFFLINE ANSWER GENERATOR] [OFFLINE LLM CALL]"
