from rag_system.infrastructure.document_loader import OfflineDocumentLoader
from rag_system.infrastructure.document_retriever_loader import DOCUMENT_RETRIVAL_CLASS_MAP


class TestDocumentRetrieverLoader:
    def test_document_retrieval_class_map_contains_expected_classes(self):
        """Test that the document retrieval class map contains the expected classes."""
        assert "offline" in DOCUMENT_RETRIVAL_CLASS_MAP
        assert DOCUMENT_RETRIVAL_CLASS_MAP["offline"] == OfflineDocumentLoader

    def test_document_retrieval_class_map_keys(self):
        """Test that the document retrieval class map has the expected keys."""
        expected_keys = {"offline"}
        assert set(DOCUMENT_RETRIVAL_CLASS_MAP.keys()) == expected_keys

    def test_document_retrieval_class_map_values_are_classes(self):
        """Test that the document retrieval class map values are classes."""
        for loader_class in DOCUMENT_RETRIVAL_CLASS_MAP.values():
            assert isinstance(loader_class, type)