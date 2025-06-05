from domain.document import Document, EmptyResponse, Documents
from abc import ABC, abstractmethod

class BaseQuestionClassifier(ABC):
    """
    Base class for question classifiers.
    This class can be extended to implement different classification strategies.
    """
    @abstractmethod
    def classify(self, question: str) -> Document| EmptyResponse:
        raise NotImplementedError("This method should be overridden by subclasses.")

class QuestionClassifier(BaseQuestionClassifier):
    """
    Classify the question and retrieve the most relevant document.
    """
    def __init__(self, documents: Documents):
        self.documents = documents

    def classify(self, question: str) -> Document | EmptyResponse:
        for doc in self.documents:
            #This must be a more sophisticated search in the future
            if question in doc["questions"]:
                return doc
        return {"content": None}