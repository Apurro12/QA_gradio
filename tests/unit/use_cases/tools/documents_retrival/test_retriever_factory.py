from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool

from rag_system.domain.document import Document, EmptyResponse
from rag_system.infrastructure.offline_document_loader import OfflineDocumentLoader
from rag_system.use_cases.tools.documents_retrival.retriever_factory import (
    RETRIEVAL_STRATEGY_CLASS_MAP,
    RETRIEVAL_STRATEGY_WKARGS_MAP,
    factory_documents_retrieval_tool,
)
from rag_system.use_cases.tools.documents_retrival.strategies.exact_match_retrieval_strategy import (
    ExactMatchRetrievalStrategy,
)
from rag_system.use_cases.tools.documents_retrival.strategies.llm_retrieval_strategy import (
    LLMRetrievalStrategy,
)


class TestRetrieverFactory:
    def test_retrieval_strategy_class_map_contains_expected_classes(self):
        """Test that the retrieval strategy class map contains the expected classes."""
        assert "exact_match" in RETRIEVAL_STRATEGY_CLASS_MAP
        assert "llm" in RETRIEVAL_STRATEGY_CLASS_MAP
        assert RETRIEVAL_STRATEGY_CLASS_MAP["exact_match"] == ExactMatchRetrievalStrategy
        assert RETRIEVAL_STRATEGY_CLASS_MAP["llm"] == LLMRetrievalStrategy

    def test_retrieval_strategy_class_map_keys(self):
        """Test that the retrieval strategy class map has the expected keys."""
        expected_keys = {"exact_match", "llm"}
        assert set(RETRIEVAL_STRATEGY_CLASS_MAP.keys()) == expected_keys

    def test_retrieval_strategy_class_map_values_are_classes(self):
        """Test that the retrieval strategy class map values are classes."""
        for strategy_class in RETRIEVAL_STRATEGY_CLASS_MAP.values():
            assert isinstance(strategy_class, type)

    @patch("rag_system.use_cases.tools.documents_retrival.retriever_factory.LLMClient")
    @patch("rag_system.use_cases.tools.documents_retrival.retriever_factory.ChatOpenAI")
    def test_retrieval_strategy_wkargs_map_with_llm(self, mock_chat_openai, mock_llm_client):
        """Test RETRIEVAL_STRATEGY_WKARGS_MAP with LLM configuration."""
        query_service_kwargs = {"llm": {"model": "gpt-3.5-turbo"}}
        
        mock_chat_openai_instance = MagicMock()
        mock_chat_openai.return_value = mock_chat_openai_instance
        
        mock_llm_client_instance = MagicMock()
        mock_llm_client.return_value = mock_llm_client_instance

        result = RETRIEVAL_STRATEGY_WKARGS_MAP(query_service_kwargs)

        mock_chat_openai.assert_called_once_with(model="gpt-3.5-turbo")
        mock_llm_client.assert_called_once()
        assert result == {"llm": mock_llm_client_instance}

    def test_retrieval_strategy_wkargs_map_with_exact_match(self):
        """Test RETRIEVAL_STRATEGY_WKARGS_MAP with exact match configuration."""
        query_service_kwargs = {"exact_match": {}}
        
        result = RETRIEVAL_STRATEGY_WKARGS_MAP(query_service_kwargs)
        
        assert result == {}

    def test_retrieval_strategy_wkargs_map_with_unsupported_config(self):
        """Test RETRIEVAL_STRATEGY_WKARGS_MAP with unsupported configuration."""
        query_service_kwargs = {"unsupported": {}}
        
        with pytest.raises(AssertionError, match="QueryServiceWargs .* is not supported"):
            RETRIEVAL_STRATEGY_WKARGS_MAP(query_service_kwargs)

    def test_factory_documents_retrieval_tool_creates_valid_tool(self):
        """Test that factory_documents_retrieval_tool creates a valid BaseTool."""
        mock_document_retriever = MagicMock(spec=OfflineDocumentLoader)
        
        tool = factory_documents_retrieval_tool(
            retrieval_strategy=ExactMatchRetrievalStrategy,
            document_retriever=mock_document_retriever
        )
        
        assert isinstance(tool, BaseTool)
        assert tool.name == "documents_retrieval_tool"

    def test_factory_documents_retrieval_tool_invocation_returns_document(self):
        """Test that the created tool can be invoked and returns a Document."""
        mock_document_retriever = MagicMock(spec=OfflineDocumentLoader)
        mock_document = Document(content="Test document", metadata={})
        
        # Mock the strategy's classify method to return a document
        with patch.object(ExactMatchRetrievalStrategy, 'classify', return_value=mock_document):
            tool = factory_documents_retrieval_tool(
                retrieval_strategy=ExactMatchRetrievalStrategy,
                document_retriever=mock_document_retriever
            )
            
            result = tool.invoke({"user_input": "test query"})
            
            assert result == mock_document

    def test_factory_documents_retrieval_tool_invocation_returns_empty_response(self):
        """Test that the created tool can return an EmptyResponse."""
        mock_document_retriever = MagicMock(spec=OfflineDocumentLoader)
        empty_response = EmptyResponse()
        
        # Mock the strategy's classify method to return an empty response
        with patch.object(ExactMatchRetrievalStrategy, 'classify', return_value=empty_response):
            tool = factory_documents_retrieval_tool(
                retrieval_strategy=ExactMatchRetrievalStrategy,
                document_retriever=mock_document_retriever
            )
            
            result = tool.invoke({"user_input": "test query"})
            
            assert result == empty_response
