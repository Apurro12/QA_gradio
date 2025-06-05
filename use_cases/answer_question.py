from domain.llm_client import BaseLLMClient
from abc import ABC, abstractmethod

class BaseAnswerGenerator(ABC):
    """
    Base class for answer generators.
    This class can be extended to implement different answer generation strategies.
    """
    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")

class AnswerGenerator(BaseAnswerGenerator):
    """
    Generate an answer to the question using the provided context.
    """
    def __init__(self, llm: BaseLLMClient):
        self.llm = llm  # Correctly assign the llm instance

    def generate(self, question: str, context: str) -> str:
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        response: str = self.llm.invoke(prompt)  # Explicitly annotate response as str
        return response