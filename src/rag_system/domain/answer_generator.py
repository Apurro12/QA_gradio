from abc import ABC, abstractmethod

class BaseAnswerGenerator(ABC):
    """
    Base class for answer generators.
    This class can be extended to implement different answer generation strategies.
    """
    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        pass # pragma: no cover