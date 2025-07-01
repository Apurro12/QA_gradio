import json

from pydantic import BaseModel

from rag_system.domain.classify_question import BaseQuestionClassifier
from rag_system.domain.conversation_manager import Message
from rag_system.domain.document import Document, EmptyResponse
from rag_system.domain.document_loader import BaseDocumentLoader
from rag_system.domain.llm_client import BaseLLMClient


class OutputSchema(BaseModel):
    """Schema for response."""

    index: int

class LLMRetrievalStrategy(BaseQuestionClassifier):
    """Retrieve documents using LLM-based strategy.
    
    This class is supposed to be instantiated with an llm that uses OutputSchema as output schema.
    """

    def __init__(
            self,
            document_loader: BaseDocumentLoader,
            llm: BaseLLMClient):
        """Initialize the retrieval strategy."""
        self.document_loader = document_loader
        self.llm = llm

        if self.llm._OutputSchema != OutputSchema: # type: ignore
            raise ValueError(
                "LLMClient must be initialized with an OutputSchema."
            )

    def classify(self, question: str) -> Document | EmptyResponse:
        """Retrieve document by selecting the best match using LLM."""
        documents: list[Document] = list(self.document_loader.load())
        if not documents:
            return EmptyResponse(content=None)

        prompt = (
            "You are a classifier. Given a user question, select the set of example "
            "questions that best represent the actual user question.\n\n"
        )
        for i, doc in enumerate(documents, 1):
            prompt += f"{i}. {doc['questions']}\n"
        prompt += "\n \n \n"
        prompt += f"User Question: {question}\n"
        prompt += (
            "Respond ONLY with the number of the best matching document. If none match,"
            "respond with 0."
        )

        # TODO:
        try:
            response = self.llm.invoke([Message(role="user", content=prompt)])
            index = int(json.loads(response)["index"])
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
    from langchain_openai import ChatOpenAI

    from rag_system.infrastructure.document_loader import OfflineDocumentLoader
    from rag_system.use_cases.llm_client import LLMClient


    llm = LLMClient(ChatOpenAI(model="gpt-4.1"), _OutputSchema = OutputSchema)
    document_loader = OfflineDocumentLoader()
    llm_classifier = LLMRetrievalStrategy(document_loader, llm)

    print(llm_classifier.classify("How can I get a refund?"))
