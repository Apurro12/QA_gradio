from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the .env file relative to the current file's directory
load_dotenv(os.path.join(current_dir, "../.env"), override=True)

# Should be this an abstract class?
# Should be this an interface?
# Should OpenAIClient inherit from the abstract class or interface?
# Is this just boilerplate code?
# To be defined in the future
class OpenAIClient:
    def __init__(self):
        """
        Initialize the LangChain ChatOpenAI client with the provided API key.
        """
        self.llm = OpenAI(temperature = 0)

    def invoke(self, prompt: str) -> str:
        """
        Generate a response using LangChain's ChatOpenAI client.
        """
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            return f"An error occurred: {e}"
        
if __name__ == "__main__":
    client = OpenAIClient()
    test_prompt = "What is the capital of France?"
    response = client.invoke(test_prompt)
    print(f"Response: {response}")