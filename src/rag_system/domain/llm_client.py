from abc import ABC, abstractmethod

### from mcp import Tool as MCPTool #TODO add MCP support later
from langchain_core.tools import BaseTool as LangchainTool 
from rag_system.domain.conversation_manager import Message
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage
from langchain_core.language_models import LanguageModelInput

class BaseLLMClient(ABC): 
    """
    Base class for LLM clients.
    This class can be extended to implement specific LLM client functionalities.
    """

    # Do not change to have [] as default value
    # It could lead to unexpected behavior due to mutable default arguments
    @abstractmethod
    def __init__(self, _base_llm: ChatOpenAI, _tools: None | List[LangchainTool] = None) -> None:
        """
        Initialize the LLM client with OpenAI client and tools.
        After initialization, tools should be loaded automatically.
        """
        if _tools:
            self._tools = _tools
        else:
            self._tools = []

        self._base_llm: ChatOpenAI  = _base_llm
        self._llm_with_tools: Runnable[LanguageModelInput, BaseMessage] = self.load_tools(self._tools)

    @abstractmethod
    def invoke(self, messages: List[Message]) -> str:
        """
        Abstract method to be implemented by subclasses to generate a response.
        """
        raise NotImplementedError("Subclasses must implement this method.") 
    
    def load_tools(self, _tools: List[LangchainTool]) -> ChatOpenAI | Runnable[LanguageModelInput, BaseMessage]:
        """
        Abstract method to load tools into the LLM client.
        """
        if self._tools:
            return self._base_llm.bind_tools(_tools, tool_choice = "auto")  # type: ignore[misc]        

        else:
            return self._base_llm