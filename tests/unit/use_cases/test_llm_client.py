from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_system.domain.conversation_manager import Message
from rag_system.use_cases.llm_client import LLMClient, OfflineLLMClient

class TestLLMClient:
    def test_llmclient_invoke(self):
        mocked_response = SimpleNamespace(content="mocked response")
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mocked_response
        
        test_prompt = [Message(role="user", content="test prompt")]

        llm = LLMClient(mock_llm_instance, [])
        result = llm.invoke(test_prompt)
        mock_llm_instance.invoke.assert_called_once_with([m.model_dump() for m in test_prompt])
        assert result == mocked_response.content
        assert llm._tools == [] # type: ignore

    def test_llmclient_error(self):
        test_prompt = [Message(role="user", content="test prompt")]
        error_message = "An error occurred: division by zero"

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke = lambda prompt: 1 / 0  # type: ignore

        llm = LLMClient(mock_llm_instance, [])
        result = llm.invoke(test_prompt)
        assert result == error_message
        assert llm._tools == [] # type: ignore


    def test_invoke_with_tools(self):
        # Write this for the online version
        mock_llm_instance = MagicMock()
        mock_llm_instance.bind_tools = lambda _tools, tool_choice: mock_llm_instance  # type: ignore
        
        mock_llm_response = "mocked response"
        mock_llm_instance.invoke.return_value = SimpleNamespace(content=mock_llm_response)

        test_prompt = [Message(role="user", content="test prompt")]
        mock_tools = MagicMock()

        mockTool1 = MagicMock()
        mockTool1.name = "Tool1"

        mockTool2 = MagicMock()
        mockTool2.name = "Tool2"

        mock_tools.__iter__.return_value = [mockTool1, mockTool2]

        llm = LLMClient(mock_llm_instance, mock_tools)
        result = llm.invoke(test_prompt)
        mock_llm_instance.invoke.assert_called_once_with([m.model_dump() for m in test_prompt])
        
        assert result == mock_llm_response
        assert llm._tools == mock_tools  # type: ignore

class TestOfflineLLMClient:
    def test_invoke(self):
        
        mock_llm_instance = MagicMock()
        test_prompt = [Message(role="user", content="test prompt")]

        llm = OfflineLLMClient(mock_llm_instance, [])
        result = llm.invoke(test_prompt)
        mock_llm_instance.invoke.assert_not_called()
        
        mocked_response = f"Last message is: '{test_prompt[0].content}'. Available tools: '[]'"
        assert result == mocked_response

    def test_invoke_with_tools(self):
        # Write this for the online version
        mock_llm_instance = MagicMock()
        mock_llm_instance.bind_tools = lambda _tools, tool_choice: mock_llm_instance  # type: ignore

        test_prompt = [Message(role="user", content="test prompt")]
        mock_tools = MagicMock()

        mockTool1 = MagicMock()
        mockTool1.name = "Tool1"

        mockTool2 = MagicMock()
        mockTool2.name = "Tool2"

        mock_tools.__iter__.return_value = [mockTool1, mockTool2]

        llm = OfflineLLMClient(mock_llm_instance, mock_tools)
        result = llm.invoke(test_prompt)
        mock_llm_instance.invoke.assert_not_called()
        
        mocked_response = f"Last message is: '{test_prompt[0].content}'. Available tools: '['Tool1', 'Tool2']'"
        assert result == mocked_response
        assert llm._tools == mock_tools  # type: ignore