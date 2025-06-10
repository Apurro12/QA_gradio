import pytest
from rag_system.domain.agent import BaseAgent
from rag_system.interfaces.strategies_classifier.exact_match_classifier import ExactMatchClassifier
from rag_system.use_cases.agent import Agent
from rag_system.use_cases.answer_generator import OfflineAnswerGenerator
from rag_system.infrastructure.document_loader import OfflineDocumentLoader
from rag_system.infrastructure.llm_client import OfflineLLMClient

## TO DO
## ADD THE CASE WHERE THE LOADED GIVES NO DOCUMENTS

@pytest.fixture
def offline_agent():
    llm = OfflineLLMClient()
    offline_document_loader = OfflineDocumentLoader()
    question_classifier = ExactMatchClassifier(document_loader=offline_document_loader)
    answer_generator = OfflineAnswerGenerator(llm=llm)
    return Agent(question_classifier, answer_generator)


def test_offline_agent_find_response(offline_agent: BaseAgent):
    response = offline_agent.respond_user_question("How can I get a refund?")
    assert isinstance(response, str)

    expected_offline_response = """offline answer generator: 
 question: 'How can I get a refund?' 
 context: 'Our refund policy allows returns within 30 days.' 
 llm call: 'llm_call'"""

    assert expected_offline_response == response

def test_offline_agent_not_find_response(offline_agent: BaseAgent):
    response = offline_agent.respond_user_question("What is the capital of France?")
    assert isinstance(response, str)

    expected_offline_response = 'No relevant information found to answer your question.'
    assert expected_offline_response == response