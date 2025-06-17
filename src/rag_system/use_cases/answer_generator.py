from rag_system.domain.answer_generator import BaseAnswerGenerator
from rag_system.domain.llm_client import BaseLLMClient

class OfflineAnswerGenerator(BaseAnswerGenerator):
    """
    Class to generate offline responses
    """

    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    def generate(self, question: str, context: str) -> str:
        return f"[OFFLINE ANSWER GENERATOR] {self.llm.invoke('[OFFLINE LLM CALL]')}"
    

class AnswerGenerator(BaseAnswerGenerator):
    """
    Generate an answer to the question using the provided context.
    """
    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    def generate(self, question: str, context: str) -> str:
        prompt = f"respond this question:\n{question}\n\nbased in this context: \n{context}\n"
        response: str = self.llm.invoke(prompt)
        return response