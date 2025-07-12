import pytest

from rag_system.domain.document import  Documents
from rag_system.domain.document_loader import BaseDocumentLoader
from rag_system.domain.document import Documents, example_docs

class TestBaseDocumentLoader:

    @pytest.fixture
    def document_loader(self):

        class ConcreteDocumentLoader(BaseDocumentLoader):
            def __init__(self):
                super().__init__()

            def load(self) -> Documents:
                return example_docs

        return ConcreteDocumentLoader()

    def test_load_offline_example_docs(self, document_loader: BaseDocumentLoader):
        """Test that a concrete implementation works correctly."""

        loaded_example_docs = document_loader.load_offline_example_docs()
        assert loaded_example_docs == example_docs