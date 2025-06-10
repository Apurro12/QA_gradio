from abc import ABC, abstractmethod
class BaseLLMClient(ABC):
    """
    Base class for LLM clients.
    This class can be extended to implement specific LLM client functionalities.
    """

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """
        Abstract method to be implemented by subclasses to generate a response.
        """
        raise NotImplementedError("Subclasses must implement this method.") # pragma: no cover