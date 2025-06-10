from abc import ABC, abstractmethod
from rag_system.domain.document import Document, EmptyResponse

class BaseQuestionClassifier(ABC):
    """
    Base class for question classifiers.
    This class can be extended to implement different classification strategies.
    """
    @abstractmethod
    def classify(self, question: str) -> Document| EmptyResponse:
        raise NotImplementedError("This method should be overridden by subclasses.") # pragma: no cover