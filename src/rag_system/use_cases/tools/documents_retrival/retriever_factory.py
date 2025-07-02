from typing import Annotated

from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI  # type: ignore

from rag_system.domain.classify_question import BaseQuestionClassifier
from rag_system.domain.document import Document, EmptyResponse
from rag_system.domain.document_loader import BaseDocumentLoader
from rag_system.use_cases.llm_client import LLMClient
from rag_system.use_cases.tools.documents_retrival.strategies.llm_retrieval_strategy import LLMRetrievalStrategy, OutputSchema
from rag_system.use_cases.tools.documents_retrival.strategies.exact_match_retrieval_strategy import ExactMatchRetrievalStrategy


def factory_documents_retrieval_tool(
    retrieval_strategy: type[BaseQuestionClassifier],
    document_retriever: BaseDocumentLoader,
    **kwargs: dict, # type: ignore
) -> BaseTool: # type: ignore

    @tool
    def documents_retrieval_tool(
        user_input: Annotated[
            str, "The description of the documents the user wants to retrieve"
        ]) -> Document | EmptyResponse:
        """Retrieve documents from source.

        This function retrieves documents based on the user's input.
        It is not necesary that the user explicitely asks for documents,
        but the system can infer that the user wants to retrieve documents
        based on their input.
        If the question is from common sense or common knowledge do not call this tool.
        """
        return retrieval_strategy(document_retriever, **kwargs).classify(user_input)

    return documents_retrieval_tool



RETRIEVAL_STRATEGY_CLASS_MAP: dict[str, type[BaseQuestionClassifier]] = {
    "exact_match": ExactMatchRetrievalStrategy,
    "llm": LLMRetrievalStrategy,
}

def RETRIEVAL_STRATEGY_WKARGS_MAP(QueryServiceWargs: dict) -> object: ## type: ignore

    if "llm" in QueryServiceWargs:
        return {"llm": LLMClient(
            ChatOpenAI(model=QueryServiceWargs["llm"]["model"]), # type: ignore
            _OutputSchema=OutputSchema
        )} 
    
    if "exact_match" in QueryServiceWargs:
        return dict() # type: ignore
    
    assert False, (
        f"QueryServiceWargs {QueryServiceWargs} is not supported."
    )

# Pre-configured tool for the main application
from rag_system.infrastructure.document_loader import OfflineDocumentLoader
from rag_system.use_cases.tools.documents_retrival.strategies.exact_match_retrieval_strategy import (
    ExactMatchRetrievalStrategy,
)

load_documents_tool = factory_documents_retrieval_tool(
    retrieval_strategy=ExactMatchRetrievalStrategy,
    document_retriever=OfflineDocumentLoader()
)

if __name__ == "__main__":
    print(load_documents_tool.invoke("What is the refund policy?"))
