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

    def test_offline_invoke_with_tool_call(self):
        """Test offline LLM client with tool call."""
        import json
        from unittest.mock import MagicMock
        
        mock_llm_instance = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.invoke.return_value = {"result": "tool response"}
        
        llm = OfflineLLMClient(mock_llm_instance, [mock_tool])
        
        # Test tool call format
        tool_call_message = Message(
            role="user", 
            content='tool: {"name":"test_tool", "args":{"input":"test"}}'
        )
        
        result = llm.invoke([tool_call_message])
        
        mock_tool.invoke.assert_called_once_with(input="test")
        assert result == json.dumps({"result": "tool response"})

    def test_offline_invoke_list_tools(self):
        """Test offline LLM client listing tools."""
        mock_llm_instance = MagicMock()
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock() 
        mock_tool2.name = "tool2"
        
        llm = OfflineLLMClient(mock_llm_instance, [mock_tool1, mock_tool2])
        
        result = llm.invoke([Message(role="user", content="list tools")])
        
        assert result == "Available tools: ['tool1', 'tool2']"

    def test_offline_invoke_with_string_input(self):
        """Test offline LLM client with string input."""
        mock_llm_instance = MagicMock()
        
        llm = OfflineLLMClient(mock_llm_instance, [])
        
        result = llm.invoke("test message")
        
        assert result == "Last message is: 'test message'. Available tools: '[]'"

    def test_llm_client_invoke_with_string_input(self):
        """Test LLM client with string input."""
        mocked_response = SimpleNamespace(content="mocked response")
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mocked_response
        
        llm = LLMClient(mock_llm_instance, [])
        result = llm.invoke("test prompt")
        
        expected_call = [{"role": "user", "content": "test prompt"}]
        mock_llm_instance.invoke.assert_called_once_with(expected_call)
        assert result == mocked_response.content

    def test_llm_client_with_tool_calls_error_handling(self):
        """Test LLM client error handling with tool calls."""
        from langchain_core.messages import AIMessage
        
        # Mock tool call response
        mock_tool_call = {
            "name": "test_tool",
            "args": {"input": "test"},
            "id": "call_123",
            "type": "function"
        }
        
        # Mock AI message with tool calls that will cause an error
        mock_ai_message = AIMessage(content="", tool_calls=[mock_tool_call])
        
        # Mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        # This will cause an error in the complex conversion logic
        mock_tool.invoke.side_effect = Exception("Tool error")
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_ai_message
        
        llm = LLMClient(mock_llm_instance, [mock_tool])
        
        test_messages = [Message(role="user", content="test prompt")]
        result = llm.invoke(test_messages)
        
        # Should catch the error and return an error message
        assert "An error occurred:" in result

    def test_llm_client_assertion_error_multiple_tools_same_name(self):
        """Test LLM client assertion error when multiple tools have the same name."""
        from langchain_core.messages import AIMessage
        
        mock_tool_call = {
            "name": "duplicate_tool",
            "args": {"input": "test"},
            "id": "call_123",
            "type": "function"
        }
        
        mock_ai_message = AIMessage(content="", tool_calls=[mock_tool_call])
        
        # Create two tools with the same name
        mock_tool1 = MagicMock()
        mock_tool1.name = "duplicate_tool"
        mock_tool2 = MagicMock()
        mock_tool2.name = "duplicate_tool"
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_ai_message
        
        llm = LLMClient(mock_llm_instance, [mock_tool1, mock_tool2])
        
        test_messages = [Message(role="user", content="test prompt")]
        
        # This should raise an assertion error due to duplicate tool names
        result = llm.invoke(test_messages)
        
        # The error should be caught and returned as a string
        assert "An error occurred:" in result

    def test_llm_client_assertion_error_no_tool_found(self):
        """Test LLM client assertion error when no tool with the specified name is found."""
        from langchain_core.messages import AIMessage
        
        mock_tool_call = {
            "name": "nonexistent_tool",
            "args": {"input": "test"},
            "id": "call_123",
            "type": "function"
        }
        
        mock_ai_message = AIMessage(content="", tool_calls=[mock_tool_call])
        
        mock_tool = MagicMock()
        mock_tool.name = "existing_tool"
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_ai_message
        
        llm = LLMClient(mock_llm_instance, [mock_tool])
        
        test_messages = [Message(role="user", content="test prompt")]
        
        # This should raise an assertion error due to tool not found
        result = llm.invoke(test_messages)
        
        # The error should be caught and returned as a string
        assert "An error occurred:" in result

    def test_llm_client_content_assertion_error(self):
        """Test LLM client assertion error when response content is not a string."""
        mock_response = SimpleNamespace(content=123)  # Non-string content
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        
        llm = LLMClient(mock_llm_instance, [])
        
        test_messages = [Message(role="user", content="test prompt")]
        result = llm.invoke(test_messages)
        
        # The assertion error should be caught and returned as a string
        assert "An error occurred:" in result

class TestLLMClientAdvanced:
    def test_llm_client_simple_response_without_tool_calls(self):
        """Test simple response path when there are no tool calls."""
        # This tests the normal path (lines 139-147) when response has no tool_calls
        mock_response = SimpleNamespace(content="Simple response")
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        
        llm = LLMClient(mock_llm_instance, [])
        
        test_messages = [Message(role="user", content="test prompt")]
        result = llm.invoke(test_messages)
        
        assert result == "Simple response"

    def test_llm_client_response_path_coverage(self):
        """Test different response paths for coverage."""
        # Test with empty tool list and normal response
        mock_response = SimpleNamespace(content="Response with empty tools")
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        
        llm = LLMClient(mock_llm_instance, [])
        
        test_messages = [Message(role="user", content="test prompt")]
        result = llm.invoke(test_messages)
        
        assert result == "Response with empty tools"