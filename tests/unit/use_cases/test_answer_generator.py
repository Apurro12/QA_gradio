from rag_system.use_cases.answer_generator import AnswerGenerator, OfflineAnswerGenerator
from rag_system.infrastructure.llm_client import OfflineLLMClient
import pytest
from unittest.mock import MagicMock

class TestOfflineAnswerGenerator:
    @pytest.fixture
    def offline_answer_generator(self):
        llm = OfflineLLMClient()
        answer_generator = OfflineAnswerGenerator(llm=llm)
        return answer_generator

    def test_generate(self, offline_answer_generator: OfflineAnswerGenerator):
        test_question = "This is a test question"
        test_response =  "This is a test context"
        offline_response = offline_answer_generator.generate(test_question, test_response)
        assert isinstance(offline_response, str)

        expected_offline_response = (
            f"offline answer generator: \n question: '{test_question}' \n context: '{test_response}' \n llm call: '{offline_answer_generator.llm.invoke('llm_call')}'"
        )
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
        context = "AI stands for Artificial Intelligence."
        result = answer_generator.generate(question, context)
        expected_prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        mock_llm.invoke.assert_called_once_with(expected_prompt)
        assert result == "mocked llm response"


