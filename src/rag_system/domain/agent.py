from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for agents.
    """
    @abstractmethod
    def respond_user_question(self, question: str) -> str:
        pass