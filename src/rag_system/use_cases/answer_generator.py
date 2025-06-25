from rag_system.domain.answer_generator import BaseAnswerGenerator
from rag_system.domain.llm_client import BaseLLMClient
from typing import List
from rag_system.domain.conversation_manager import Message


class OfflineAnswerGenerator(BaseAnswerGenerator):
    """
    Class to generate offline responses
    """

    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    def generate(self, messages: List[Message]) -> str:
        return f"[OFFLINE ANSWER GENERATOR], last message: {str(messages[-1])}"


class AnswerGenerator(BaseAnswerGenerator):
    """
    Generate an answer to the question using the provided context.
    """
    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    def generate(self, messages: List[Message]) -> str:
        
        #TODO think how to pass prompt templates, or what is necesary to generate the prompt
        #prompt = f"respond this question:\n{question}\n\nbased in this context: \n{context}\n"
        response: str = self.llm.invoke(messages)
        return response