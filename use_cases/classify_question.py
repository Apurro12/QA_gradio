from docs.example_docs import Document
from docs.example_docs import documents


class BaseQuestionClassifier:
    """
    Base class for question classifiers.
    This class can be extended to implement different classification strategies.
    """
    def classify(self, question: str) -> Document:
        raise NotImplementedError("This method should be overridden by subclasses.")

class QuestionClassifier(BaseQuestionClassifier):
    """
    Classify the question and retrieve the most relevant document.
    """
    def __init__(self):
        self.documents = documents

    def classify(self, question: str) -> Document:
        for doc in documents:
            #This must be a more sophisticated search in the future
            if question in doc["questions"]:
                return doc
        return {"content": "No relevant document found.", "questions": []}