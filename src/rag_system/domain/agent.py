from abc import ABC, abstractmethod

from rag_system.domain.conversation_manager import Message


class BaseAgent(ABC):
    """respond_user_question: respond just one question.

    chat: respond and use history
    """

    @abstractmethod
    def chat(self, message: str, history: list[Message], session_id: str | None = None) -> str:
        """Chat with the user, using the history of messages."""
        pass  # pragma: no cover
