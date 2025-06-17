from typing import List
from rag_system.domain.conversation_manager import BaseConversationManager, Message, MessageWithTimestamp
from datetime import datetime

class InMemoryConversationManager(BaseConversationManager):
    def __init__(self) -> None:
        self.messages: List[MessageWithTimestamp] = []
    
    def add_message(self, message: Message) -> None:
        message_with_timestamp = MessageWithTimestamp(**message.model_dump(), timestamp=datetime.timestamp(datetime.now()))
        self.messages.append(message_with_timestamp)
    
    def get_conversation_history(self) -> List[MessageWithTimestamp]:
        return self.messages.copy()
    
    def clear_conversation(self) -> None:
        self.messages.clear()