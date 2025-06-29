from rag_system.domain.classify_question import BaseQuestionClassifier
from rag_system.domain.document import Document, EmptyResponse
from rag_system.domain.document_loader import BaseDocumentLoader
from rag_system.domain.llm_client import BaseLLMClient


class LLMClassifier(BaseQuestionClassifier):
    """Classify the question."""

    def __init__(self, document_loader: BaseDocumentLoader, llm: BaseLLMClient):
        """Initialize the classifier."""
        self.document_loader = document_loader
        self.llm = llm

    def classify(self, question: str) -> Document | EmptyResponse:
        """Classify the question by selecting the best match."""
        documents: list[Document] = list(self.document_loader.load())
        if not documents:
            return EmptyResponse(content=None)

        prompt = (
            "You are a classifier. Given a user question, select the set of example "
            "questions that best represent the actual user question.\n\n"
        )
        for i, doc in enumerate(documents, 1):
            prompt += f"{i}. {doc["questions"]}\n"
        prompt += "\n \n \n"
        prompt += f"User Question: {question}\n"
        prompt += (
            "Respond ONLY with the number of the best matching document. If none match,"
            "respond with 0."
        )

        # TODO:
        try:
            response = self.llm.invoke(prompt)
            index = int(response)
        except (ValueError, TypeError) as err:
            raise ValueError(
                """
                LLM response should be an integer representing
                the index of the best matching document.
                """
            ) from err

        if index == 0:
            return EmptyResponse(content=None)
        return documents[index - 1]


if __name__ == "__main__":  # pragma: no cover
    from rag_system.infrastructure.document_loader import OfflineDocumentLoader
    from rag_system.use_cases.llm_client import OfflineLLMClient

    llm = OfflineLLMClient()
    document_loader = OfflineDocumentLoader()
    llm_classifier = LLMClassifier(document_loader, llm)

    llm_classifier.classify("How can I get a refund?")
