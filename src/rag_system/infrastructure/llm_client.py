from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import List
from rag_system.domain.llm_client import BaseLLMClient
from rag_system.domain.conversation_manager import Message, ToolCallMessage, ToolResponseMessage, ToolResponseMessageOpenAI
from langchain_core.tools import BaseTool as LangchainTool
from langchain_core.messages import AIMessage
from langchain_core.messages import convert_to_openai_messages # type: ignore[misc]

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the .env file relative to the current file's directory
load_dotenv(os.path.join(current_dir, "../../../.env"), override=True)

class OfflineLLMClient(BaseLLMClient):
    def __init__(self, _base_llm: ChatOpenAI, _tools: List[LangchainTool]) -> None:
        super().__init__(_base_llm, _tools)

    def invoke(self, messages: List[Message]) -> str:
        return str(messages)

# TODO: Here the conversation manager should be able to save the internal conversation state
class LLMClient(BaseLLMClient):
    def __init__(self, _base_llm: ChatOpenAI, _tools: None | List[LangchainTool] = None) -> None:
        """
        Initialize the LangChain ChatOpenAI client with the provided API key and tools.
        """
        super().__init__(_base_llm, _tools)

    def invoke(self, messages: List[Message]) -> str:
        """
        Generate a response using LangChain's ChatOpenAI client.
        """
        try:

            # Convert Pydantic Message objects to dicts as expected by ChatOpenAI
            messages_openai_format = [m.model_dump(include={"role", "content"}) for m in messages]
            response = self._llm_with_tools.invoke(
                messages_openai_format)


            if isinstance(response, AIMessage) and len(response.tool_calls) > 0:
                
                # Just to check format
                tool_call_message = ToolCallMessage(**convert_to_openai_messages(response)).model_dump() # type: ignore
                tool_call_responses: List[ToolResponseMessageOpenAI] = []
                for tool_call in response.tool_calls:

                    # Call the tool with the arguments provided in the tool call
                    expected_desired_tool = list(filter(lambda row:  row.name == tool_call["name"], self._tools))
                    assert len(expected_desired_tool) == 1, f"Error in tool calling, {tool_call['name']} have multiple tools with the same name or no tool with this name found"


                    desired_tool = expected_desired_tool[0]
                    tool_call_response = convert_to_openai_messages(desired_tool.invoke(tool_call)) # type: ignore
                    ToolResponseMessage(**tool_call_response) # type: ignore #Just checking type

                    tool_call_responses.append(tool_call_response) # type: ignore

                    final_message = messages_openai_format + [tool_call_message]  + tool_call_responses
                    response_with_tool_call = self._llm_with_tools.invoke(final_message)  # type: ignore
                    content = response_with_tool_call.content  # type: ignore
                    assert isinstance(content, str)
                    return content

            # Handle the content type properly - LangChain response content can be str or complex content
            # Use type: ignore to suppress the "partially unknown" type warning
            content = response.content  # type: ignore[misc]

            # TODO, add a logging here to warn that the return is not a simple string   
            assert isinstance(content, str), f"{type(content)} is not a string, but a complex content type. Please handle this case properly." # type: ignore[misc]
            return content

                
        except Exception as e:
            return f"An error occurred: {e}"

# TODO: Add this main with tool calling and check that don't call wikipedia or other not explicitly called tools
# TODO: is calling tools in the second message
if __name__ == "__main__": # pragma: no cover, JUST DO WHEN RUNNING THIS FILE DIRECTLY()
    from rag_system.infrastructure.tools.document_loader import load_documents

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
    )
    
    # Initialize LLMClient with required parameters
    client = LLMClient(_base_llm=llm, _tools=None)
    test_messages = [Message(role="user", content="What is the capital of France?")]
    response = client.invoke(test_messages)
    print(response)


    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
    )

    # TODO: Here the model should not call the tool, just return the text response
    client = LLMClient(_base_llm=llm, _tools=[load_documents])
    test_messages = [Message(role="user", content="What is the capital of France?")]
    response = client.invoke(test_messages)
    print(response)

    # test that here the model should call the tool
    test_messages = [Message(role="user", content="Give me some documents about France.")]
    response = client.invoke(test_messages)
    print(response)