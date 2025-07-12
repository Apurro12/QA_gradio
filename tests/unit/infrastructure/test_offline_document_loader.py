from rag_system.infrastructure.offline_document_loader import OfflineDocumentLoader
from rag_system.domain.document import example_docs

class TestOfflineDocumentLoader:
    def test_document_retrieval_class_map_keys(self):
        loader = OfflineDocumentLoader()

        loader_example_docs = loader.load()
        assert loader_example_docs == example_docs