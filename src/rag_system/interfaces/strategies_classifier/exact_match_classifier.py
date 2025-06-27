from rag_system.domain.classify_question import BaseQuestionClassifier
from rag_system.domain.document import Document, EmptyResponse
from rag_system.domain.document_loader import BaseDocumentLoader


class ExactMatchClassifier(BaseQuestionClassifier):
    """Classify the question and retrieve the most relevant document."""

    def __init__(self, document_loader: BaseDocumentLoader):
        """Initialize the classifier with a document loader."""
        self.document_loader = document_loader

    def classify(self, question: str) -> Document | EmptyResponse:
        """Classify the question by finding an exact match in the example questions."""
        for doc in self.document_loader.load():
            # This must be a more sophisticated search in the future
            if question in doc["questions"]:
                return doc
        return {"content": None}
