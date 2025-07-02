from unittest.mock import MagicMock

import pytest

from rag_system.domain.conversation_manager import Message
from rag_system.use_cases.llm_client import OfflineLLMClient
from rag_system.use_cases.answer_generator import (
    AnswerGenerator,
    OfflineAnswerGenerator,
)

from langchain_openai import ChatOpenAI

class TestOfflineAnswerGenerator:
    @pytest.fixture
    def offline_answer_generator(self):
        llm = OfflineLLMClient(ChatOpenAI())
        answer_generator = OfflineAnswerGenerator(llm=llm)
        return answer_generator

    def test_generate(self, offline_answer_generator: OfflineAnswerGenerator):
        test_question = "This is a test question"
        message = Message(role="user", content=test_question)
        offline_response = offline_answer_generator.generate([message])
        assert isinstance(offline_response, str)

        expected_offline_response = f"[OFFLINE ANSWER GENERATOR], last message: {message}"
        assert offline_response == expected_offline_response


class TestAnswerGenerator:
    @pytest.fixture
    def mock_llm(self):
        mock = MagicMock()
        mock.invoke.return_value = "mocked llm response"
        return mock

    @pytest.fixture
    def answer_generator(self, mock_llm: MagicMock):
        return AnswerGenerator(llm=mock_llm)

    def test_generate(self, answer_generator: AnswerGenerator, mock_llm: MagicMock):
        question = "What is AI?"
        message = Message(role="user", content=question)
        result = answer_generator.generate([message])
        mock_llm.invoke.assert_called_once_with([message])
        assert result == "mocked llm response"
