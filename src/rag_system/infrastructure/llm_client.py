from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import List
from rag_system.domain.llm_client import BaseLLMClient
from rag_system.domain.conversation_manager import Message


# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the .env file relative to the current file's directory
load_dotenv(os.path.join(current_dir, "../../../.env"), override=True)

class OfflineLLMClient(BaseLLMClient):
    def __init__(self) -> None:
        pass

    def invoke(self, messages: List[Message]) -> str:
        return str(messages)

class LLMClient(BaseLLMClient):
    def __init__(self):
        """
        Initialize the LangChain ChatOpenAI client with the provided API key.
        """
        self.llm = ChatOpenAI()

    def invoke(self, messages: List[Message]) -> str:
        """
        Generate a response using LangChain's ChatOpenAI client.
        """
        try:
            # Convert Pydantic Message objects to dicts as expected by ChatOpenAI
            messages_openai_format = [m.model_dump(include={"role", "content"}) for m in messages]
            response = self.llm.invoke(messages_openai_format)

            # Handle the content type properly - LangChain response content can be str or complex content
            # Use type: ignore to suppress the "partially unknown" type warning
            content = response.content  # type: ignore[misc]

            # TODO, add a logging here to warn that the return is not a simple string   
            assert isinstance(content, str), f"{type(content)} is not a string, but a complex content type. Please handle this case properly." # type: ignore[misc]
            return content

                
        except Exception as e:
            return f"An error occurred: {e}"
        
if __name__ == "__main__": # pragma: no cover, JUST DO WHEN RUNNING THIS FILE DIRECTLY
    client = LLMClient()
    test_messages = [Message(role="user", content="What is the capital of France?")]
    response = client.invoke(test_messages)
    print(response)