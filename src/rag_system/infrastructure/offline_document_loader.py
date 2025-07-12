from rag_system.domain.document import Documents
from rag_system.domain.document_loader import BaseDocumentLoader


class OfflineDocumentLoader(BaseDocumentLoader):
    """Load the example documents from the repo."""

    def __init__(self) -> None:
        """Initialize the offline document loader."""
        pass

    def load(self) -> Documents:
        """Load the example documents."""
        return self.load_offline_example_docs()
