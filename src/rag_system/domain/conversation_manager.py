from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Literal, List, Any, TypedDict


Role = Literal["user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class GradioHistoryMessage(TypedDict):
    role: Role
    content: str
    metadata: Any  # Gradio Response, I dontt know what this is, Check this in gradio documentation
    options: Any  # Gradio Response, I dontt know what this is, Check this in gradio documentation

class MessageWithTimestamp(Message):
    timestamp: float


class BaseConversationManager(ABC): #pragma: no cover
    @abstractmethod
    def add_message(self, message: Message) -> None:
        pass
    
    @abstractmethod
    def get_conversation_history(self) -> List[MessageWithTimestamp]:
        pass
    
    @abstractmethod
    def clear_conversation(self) -> None:
        pass