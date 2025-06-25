from abc import ABC, abstractmethod
from rag_system.domain.conversation_manager import Message
from typing import List

class BaseLLMClient(ABC):
    """
    Base class for LLM clients.
    This class can be extended to implement specific LLM client functionalities.
    """

    @abstractmethod
    def invoke(self, messages: List[Message]) -> str:
        """
        Abstract method to be implemented by subclasses to generate a response.
        """
        raise NotImplementedError("Subclasses must implement this method.") # pragma: no cover