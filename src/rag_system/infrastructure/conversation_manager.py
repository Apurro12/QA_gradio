from datetime import datetime

from rag_system.domain.conversation_manager import (
    BaseConversationManager,
    Message,
    MessageWithTimestamp,
    ToolCallMessage,
    ToolResponseMessage,
)


class InMemoryConversationManager(BaseConversationManager):
    def __init__(self) -> None:
        """Initialize the in-memory conversation manager."""
        # Storage 1: Gradio conversation (user/assistant messages)
        self.gradio_messages: list[MessageWithTimestamp] = []
        # Storage 2: Internal LLM history (tool calls and responses)
        self.internal_llm_history: list[ToolCallMessage | ToolResponseMessage] = []

    def add_message(self, message: Message) -> None:
        """Add user/assistant message to Gradio conversation history."""
        message_with_timestamp = MessageWithTimestamp(
            **message.model_dump(), timestamp=datetime.timestamp(datetime.now())
        )
        self.gradio_messages.append(message_with_timestamp)

    def add_tool_call_message(self, message: ToolCallMessage) -> None:
        """Add tool call message to internal LLM history."""
        self.internal_llm_history.append(message)

    def add_tool_response_message(self, message: ToolResponseMessage) -> None:
        """Add tool response message to internal LLM history."""
        self.internal_llm_history.append(message)

    def update_conversation(self, messages: list[Message]) -> None:
        """Update/replace the conversation with a new list of messages from Gradio."""
        # This is unnecesary, but I want to be explicit
        self.clear_conversation()
        # TODO: the timstamp should be the message time, to update later
        self.gradio_messages = [
            MessageWithTimestamp(**msg.model_dump(), timestamp=datetime.timestamp(datetime.now()))
            for msg in messages
        ]

    def get_conversation_history(self) -> list[MessageWithTimestamp]:
        """Get Gradio conversation history (user/assistant messages only)."""
        return self.gradio_messages.copy()

    def clear_conversation(self) -> None:
        """Clear Gradio conversation history."""
        self.gradio_messages.clear()

    def get_internal_llm_history(
        self,
    ) -> list[ToolCallMessage | ToolResponseMessage]:
        """Get internal LLM history (tool calls and responses)."""
        return self.internal_llm_history.copy()

    def clear_internal_llm_history(self) -> None:
        """Clear internal LLM history."""
        self.internal_llm_history.clear()
