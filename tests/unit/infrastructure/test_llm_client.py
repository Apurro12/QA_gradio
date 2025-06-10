from unittest.mock import patch, MagicMock
from rag_system.infrastructure.llm_client import LLMClient

class TestLLMClient:

    def test_llmclient_invoke(self):
        with patch("rag_system.infrastructure.llm_client.OpenAI") as MockOpenAI:
            mocked_response = "mocked response"
            test_prompt = "test prompt"

            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke.return_value = mocked_response
            MockOpenAI.return_value = mock_llm_instance

            llm = LLMClient()
            result = llm.invoke(test_prompt)
            mock_llm_instance.invoke.assert_called_once_with(test_prompt)
            assert result == mocked_response

        
    def test_llmclient_error(self):
        with patch("rag_system.infrastructure.llm_client.OpenAI") as MockOpenAI:
            test_prompt = "test prompt"
            error_message = "An error occurred: division by zero"

            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke = lambda prompt: 1/0  # type: ignore
            MockOpenAI.return_value = mock_llm_instance

            llm = LLMClient()
            result = llm.invoke(test_prompt)
            assert result == error_message