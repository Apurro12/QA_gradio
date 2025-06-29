from abc import ABC, abstractmethod
from typing import Any, Literal, TypedDict

from pydantic import BaseModel

Role = Literal["user", "assistant", "system"]


class FunctionCall(TypedDict):
    name: str
    arguments: str  # JSON string of arguments e.g. '{"city": "San Francisco"}'.
    # It is the string with the ' it is not a dict


class ToolCalls(TypedDict):
    type: Literal["function"]
    id: str
    function: FunctionCall


# This is doing deep checking into ToolCalls
# e.g. if name is an int it will fail
class ToolCallMessage(BaseModel):
    role: Literal["assistant"]
    content: Literal[""]  # OpenAI does not use content in tool calls
    tool_calls: list[ToolCalls]


class ToolCallMessageOpenAI(TypedDict):
    role: Literal["assistant"]
    content: Literal[""]  # OpenAI does not use content in tool calls
    tool_calls: list[ToolCalls]


class ToolResponseMessage(BaseModel):
    role: Literal["tool"]
    name: str
    content: str
    tool_call_id: str  # This is the id of the tool call, it is used to link the
    # response with the tool call


class ToolResponseMessageOpenAI(TypedDict):
    role: Literal["tool"]
    name: str
    content: str
    tool_call_id: str


class Message(BaseModel):
    role: Role
    content: str


class GradioHistoryMessage(TypedDict):
    role: Role
    content: str
    metadata: Any  # Gradio Response, I dontt know what this is.
    # Check this in gradio documentation
    options: Any  # Gradio Response, I dontt know what this is.
    # Check this in gradio documentation


class MessageWithTimestamp(Message):
    timestamp: float


class BaseConversationManager(ABC):  # pragma: no cover
    @abstractmethod
    def add_message(self, message: Message) -> None:
        """Add user/assistant message to Gradio conversation history."""
        pass

    @abstractmethod
    def update_conversation(self, messages: list[Message]) -> None:
        """Update/replace the conversation with a new list of messages from Gradio."""
        pass

    @abstractmethod
    def get_conversation_history(self) -> list[MessageWithTimestamp]:
        """Get Gradio conversation history (user/assistant messages only)."""
        pass

    @abstractmethod
    def clear_conversation(self) -> None:
        """Clear Gradio conversation history."""
        pass

    @abstractmethod
    def add_tool_call_message(self, message: ToolCallMessage) -> None:
        """Add tool call message to internal LLM history."""
        pass

    @abstractmethod
    def add_tool_response_message(self, message: ToolResponseMessage) -> None:
        """Add tool response message to internal LLM history."""
        pass

    @abstractmethod
    def get_internal_llm_history(
        self,
    ) -> list[ToolCallMessage | ToolResponseMessage]:
        """Get internal LLM history (tool calls and responses)."""
        pass

    @abstractmethod
    def clear_internal_llm_history(self) -> None:
        """Clear internal LLM history."""
        pass
