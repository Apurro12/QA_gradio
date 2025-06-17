from abc import ABC, abstractmethod
from typing import List

from rag_system.domain.conversation_manager import Message

class BaseAgent(ABC):
    """
    respond_user_question: respond just one question 
    chat: respond and use history
    """
    @abstractmethod
    def respond_user_question(self, question: str, history: List[Message]) -> str:
        pass # pragma: no cover

    @abstractmethod
    def chat(self, message: str, history: List[Message]) -> str:
        pass # pragma: no cover