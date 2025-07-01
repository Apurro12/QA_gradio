from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

### from mcp import Tool as MCPTool #TODO add MCP support later
from langchain_core.tools import BaseTool as LangchainTool
from pydantic import BaseModel

from rag_system.domain.conversation_manager import Message


class BaseLLMClient(ABC):
    """Base class for LLM clients.

    This class can be extended to implement specific LLM client functionalities.
    """

    # Do not change to have [] as default value
    # It could lead to unexpected behavior due to mutable default arguments
    @abstractmethod
    def __init__(
        self,
        _base_llm: BaseChatModel,
        _tools: None | list[LangchainTool] = None,
        _OutputSchema: type[BaseModel] | None = None,
    ) -> None:
        """Initialize the LLM client with OpenAI client and tools.

        After initialization, tools should be loaded automatically.
        """
        self._tools: list[LangchainTool] = _tools if _tools is not None else []
        self._OutputSchema = _OutputSchema
        self._base_llm: BaseChatModel = _base_llm
        self._llm_with_tools: Runnable[LanguageModelInput, BaseMessage] = self.load_response_format(_OutputSchema)
        self._llm_with_tools: Runnable[LanguageModelInput, BaseMessage] = (
            self.load_tools(self._tools)
        )

        # TODO: ad this the testing


    @abstractmethod
    def invoke(self, messages: list[Message]) -> str:
        """Abstract method to be implemented by subclasses to generate a response."""
        raise NotImplementedError("Subclasses must implement this method.")

    # TODO: add this the testing
    def load_response_format(self, _OutputSchema: type[BaseModel] | None) -> Runnable[LanguageModelInput, BaseMessage]:
        """Load the response format into the LLM client."""
        return self._base_llm if _OutputSchema is None else self._base_llm.bind(response_format=_OutputSchema)

    def load_tools(
        self, _tools: list[LangchainTool]
    ) -> BaseChatModel | Runnable[LanguageModelInput, BaseMessage]:
        """Abstract method to load tools into the LLM client."""
        if self._tools:
            return self._llm_with_tools.bind_tools(_tools, tool_choice="auto")  # type: ignore[misc]

        else:
            return self._llm_with_tools
