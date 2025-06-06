from rag_system.domain.answer_generator import BaseAnswerGenerator
from rag_system.domain.llm_client import BaseLLMClient

class AnswerGenerator(BaseAnswerGenerator):
    """
    Generate an answer to the question using the provided context.
    """
    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    def generate(self, question: str, context: str) -> str:
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        response: str = self.llm.invoke(prompt)
        return response