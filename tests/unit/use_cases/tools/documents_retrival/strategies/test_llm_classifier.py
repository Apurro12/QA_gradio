from unittest.mock import MagicMock

import pytest

from rag_system.domain.document import EmptyResponse, example_docs
from rag_system.infrastructure.document_loader import OfflineDocumentLoader
from rag_system.use_cases.tools.documents_retrival.strategies.llm_retrieval_strategy import LLMRetrievalStrategy, OutputSchema


class TestLLMClassifier:
    def test_llmclassifier_returns_document(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"index": 1}'
        mock_llm._OutputSchema = OutputSchema
        
        document_loader = OfflineDocumentLoader()
        llm_classifier = LLMRetrievalStrategy(document_loader, mock_llm)

        document = llm_classifier.classify("example prompt 1")
        assert document == example_docs[0]

    def test_llmclassifier_no_returns_document(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"index": 0}'
        mock_llm._OutputSchema = OutputSchema

        document_loader = OfflineDocumentLoader()
        llm_classifier = LLMRetrievalStrategy(document_loader, mock_llm)

        document = llm_classifier.classify("example prompt 1")
        assert document == EmptyResponse({"content": None})

    def test_llmclassifier_error(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "not a number"
        mock_llm._OutputSchema = OutputSchema
        # This simulates an error in the LLM response
        document_loader = OfflineDocumentLoader()
        llm_classifier = LLMRetrievalStrategy(document_loader, mock_llm)

        with pytest.raises(
            ValueError,
            match="LLM response should be an integer representing the index of the best matching document.",
        ):
            llm_classifier.classify("example prompt 1")

    def test_llm_with_no_docs(self):
        mock_llm = MagicMock()
        mock_llm._OutputSchema = OutputSchema

        mock_doc_loader = MagicMock()
        mock_doc_loader.load.return_value = []
        llm_classifier = LLMRetrievalStrategy(mock_doc_loader, mock_llm)

        response = llm_classifier.classify("example prompt")
        assert response == EmptyResponse({"content": None})