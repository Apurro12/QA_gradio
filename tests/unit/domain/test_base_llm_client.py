from unittest.mock import MagicMock
import pytest

from rag_system.domain.llm_client import BaseLLMClient
from rag_system.domain.conversation_manager import Message


class TestBaseLLMClientDomain:
    def test_base_llm_client_is_abstract(self):
        """Test that BaseLLMClient cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseLLMClient(None, None, None)  # type: ignore

    def test_base_llm_client_abstract_method(self):
        """Test that invoke method is abstract."""
        class IncompleteLLMClient(BaseLLMClient):
            pass
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteLLMClient(None, None, None)  # type: ignore

    def test_base_llm_client_concrete_implementation(self):
        """Test that a concrete implementation works correctly."""
        class ConcreteLLMClient(BaseLLMClient):
            def __init__(self, base_llm, tools, output_schema):
                super().__init__(base_llm, tools, output_schema)
            
            def invoke(self, messages: list[Message] | str) -> str:
                return "test response"
        
        client = ConcreteLLMClient(None, None, None)
        result = client.invoke("test message")
        assert result == "test response"

    def test_base_llm_client_initialization(self):
        """Test that BaseLLMClient stores initialization parameters correctly."""
        class ConcreteLLMClient(BaseLLMClient):
            def __init__(self, base_llm, tools, output_schema):
                super().__init__(base_llm, tools, output_schema)
            
            def invoke(self, messages: list[Message] | str) -> str:
                return "test response"

        mock_llm = MagicMock()
        mock_llm.bind = lambda response_format: mock_llm  # Mocking bind method
        mock_tools = ["tool1", "tool2"]
        mock_schema = "mock_schema"
        
        client = ConcreteLLMClient(mock_llm, mock_tools, mock_schema)
        
        assert client._base_llm == mock_llm
        assert client._tools == mock_tools
        assert client._OutputSchema == mock_schema

