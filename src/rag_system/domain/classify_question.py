from abc import ABC, abstractmethod

from rag_system.domain.document import Document, EmptyResponse
from rag_system.domain.document_loader import BaseDocumentLoader


class BaseQuestionClassifier(ABC):
    """Base class for question classifiers.

    This class can be extended to implement different classification strategies.
    """

    @abstractmethod
    def __init__(self, document_loader: BaseDocumentLoader):
        """Initialize the classifier."""
        pass

    @abstractmethod
    def classify(self, question: str) -> Document | EmptyResponse:
        """Classify the question and return the most relevant document."""
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )  # pragma: no cover
