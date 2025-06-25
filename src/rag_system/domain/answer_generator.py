from abc import ABC, abstractmethod
from typing import List
from rag_system.domain.conversation_manager import Message

class BaseAnswerGenerator(ABC):
    """
    Base class for answer generators.
    This class can be extended to implement different answer generation strategies.
    """
    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        pass # pragma: no cover