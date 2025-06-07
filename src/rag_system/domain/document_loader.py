from abc import ABC, abstractmethod
from rag_system.domain.document import Documents, example_docs

class BaseDocumentLoader(ABC):
    @abstractmethod
    def load(self) -> Documents:
        """Load documents from a source."""
        pass

    def load_offline_example_docs(self) -> Documents:
        """Load example documents (shared by all subclasses)."""
        return example_docs