from rag_system.domain.classify_question import BaseQuestionClassifier
from rag_system.domain.document import Document, EmptyResponse
from rag_system.domain.document_loader import BaseDocumentLoader
from rag_system.domain.llm_client import BaseLLMClient


class LLMClassifier(BaseQuestionClassifier):
    """
    Classify the question using a language model to semantically match the user question to the most relevant document.
    """
    def __init__(self, document_loader: BaseDocumentLoader, llm: BaseLLMClient):
        self.document_loader = document_loader
        self.llm = llm

    def classify(self, question: str) -> Document | EmptyResponse:
        documents = list(self.document_loader.load())
        if not documents:
            return EmptyResponse({"content": None})
        # Prepare prompt for LLM to select the best matching document
        prompt = (
            "Given the following example questions and user question, select the set of example questions that best represent the actual user question.\n\n"
            "Example questions:\n"
        )
        for idx, doc in enumerate(documents):
            prompt += f"example questions {idx+1}: {doc['questions']}\n"
        prompt += "\n \n \n"
        prompt += f"User Question: {question}\n"
        prompt += "Respond ONLY with the number of the best matching document. If none match, respond with 0."

        # TODO:
        # force the system to respond with a number, this is fragile
        response = self.llm.invoke(prompt)
        try:
            index = int(response)
        except Exception:
            assert False, "LLM response should be an integer representing the index of the best matching document."
        
        if index:
            return documents[index - 1]
        
        return EmptyResponse({"content": None})
    
if __name__ == "__main__": # pragma: no cover
        from rag_system.infrastructure.document_loader import OfflineDocumentLoader
        from rag_system.infrastructure.llm_client import OfflineLLMClient

        llm = OfflineLLMClient()
        document_loader = OfflineDocumentLoader()
        llm_classifier = LLMClassifier(document_loader, llm)

        llm_classifier.classify("How can I get a refund?")