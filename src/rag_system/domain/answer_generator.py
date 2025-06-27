from abc import ABC, abstractmethod

from rag_system.domain.conversation_manager import Message


class BaseAnswerGenerator(ABC):
    """Base class for answer generators.

    This class can be extended to implement different answer generation strategies.
    """

    @abstractmethod
    def generate(self, messages: list[Message]) -> str:
        """Generate an answer based on the provided messages."""
        pass  # pragma: no cover
