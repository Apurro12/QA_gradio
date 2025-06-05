from domain.classify_question import BaseQuestionClassifier
from domain.document import Document, EmptyResponse, Documents

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