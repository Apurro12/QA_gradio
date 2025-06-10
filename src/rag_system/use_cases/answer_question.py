from rag_system.domain.answer_generator import BaseAnswerGenerator
from rag_system.domain.llm_client import BaseLLMClient

class OfflineAnswerGenerator(BaseAnswerGenerator):
    """
    Class to generate offline responses
    """

    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    def generate(self, question: str, context: str) -> str:
        return f"offline answer generator: \n question: '{question}' \n context: '{context}' \n llm call: '{self.llm.invoke('llm_call')}'"
    

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