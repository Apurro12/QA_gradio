from rag_system.domain.document import EmptyResponse, example_docs
from rag_system.infrastructure.document_loader import OfflineDocumentLoader
from rag_system.use_cases.tools.documents_retrival.strategies.exact_match_retrieval_strategy import ExactMatchRetrievalStrategy



class TestExactMatchClassifier:
    def test_classifier_returns_document(self):
        document_loader = OfflineDocumentLoader()
        llm_classifier = ExactMatchRetrievalStrategy(document_loader)

        document = llm_classifier.classify(example_docs[0]["questions"][0])
        assert document == example_docs[0]

    def test_classifier_no_returns_no_document(self):
        document_loader = OfflineDocumentLoader()
        llm_classifier = ExactMatchRetrievalStrategy(document_loader)

        document = llm_classifier.classify("some random question")
        assert document == EmptyResponse({"content": None})
