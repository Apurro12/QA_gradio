from rag_system.domain.answer_generator import BaseAnswerGenerator
from rag_system.domain.conversation_manager import Message
from rag_system.domain.llm_client import BaseLLMClient


class OfflineAnswerGenerator(BaseAnswerGenerator):
    """Class to generate offline responses."""

    def __init__(self, llm: BaseLLMClient):
        """Initialize the answer generator with a language model client."""
        self.llm = llm

    def generate(self, messages: list[Message]) -> str:
        """Generates an answer to the question based on the provided context."""
        return f"[OFFLINE ANSWER GENERATOR], last message: {str(messages[-1])}"


class AnswerGenerator(BaseAnswerGenerator):
    """Generate an answer to the question using the provided context."""

    def __init__(self, llm: BaseLLMClient):
        """Initialize the answer generator with a language model client."""
        self.llm = llm

    def generate(self, messages: list[Message]) -> str:
        """Generates an answer to the question based on the provided context."""
        # TODO think how to pass prompt templates,
        # or what is necesary to generate the prompt
        # prompt = (
        #     f"respond this question:\n{question}\n\n"
        #     f"based in this context: \n{context}\n"
        # )
        response: str = self.llm.invoke(messages)
        return response
