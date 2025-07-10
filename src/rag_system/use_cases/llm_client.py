import json
import os
from typing import cast

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    convert_to_openai_messages,  # type: ignore[misc]
)
from langchain_core.tools import BaseTool as LangchainTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from rag_system.domain.conversation_manager import (
    Message,
    ToolCallMessage,
    ToolCallMessageOpenAI,
    ToolResponseMessage,
    ToolResponseMessageOpenAI,
)
from rag_system.domain.llm_client import BaseLLMClient

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the .env file relative to the current file's directory
load_dotenv(os.path.join(current_dir, "../../../.env"), override=True)


class OfflineLLMClient(BaseLLMClient):
    def __init__(
        self,
        _base_llm: BaseChatModel,
        _tools: list[LangchainTool] | None = None,
        _OutputSchema: type[BaseModel] | None = None,
    ) -> None:
        """Initialize the offline LLM client with the provided base LLM and tools."""
        super().__init__(_base_llm, _tools, _OutputSchema)

    def invoke(self, messages: list[Message] | str) -> str:
        """Generate a response using the offline LLM client."""
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        if messages[-1].content[:5] == "tool:":
            # e.g.
            # tool: {"name":"documents_retrieval_tool", "args":{"input":"some"}}
            # tool: {"name":"documents_retrieval_tool", "args":{"input":"How can I get a refund?"}}

            tool_call = json.loads(messages[-1].content[5:])
            tool = list(  # type: ignore
                filter(lambda row: row.name == tool_call["name"], self._tools)  # type: ignore
            )[0]
            tool_response = tool.invoke(**tool_call["args"])  # type: ignore
            return json.dumps(tool_response)

        if messages[-1].content == "list tools":
            return f"Available tools: {list(map(lambda row: row.name, self._tools))}"

        return (
            f"Last message is: '{messages[-1].content}'"
            f". Available tools: '{list(map(lambda row: row.name, self._tools))}'"
        )


# TODO: Here the conversation manager should be able to save the internal
# conversation state
class LLMClient(BaseLLMClient):
    def __init__(
        self,
        _base_llm: BaseChatModel,
        _tools: None | list[LangchainTool] = None,
        _OutputSchema: type[BaseModel] | None = None,
    ) -> None:
        """Initialize the LangChain ChatOpenAI."""
        super().__init__(_base_llm, _tools, _OutputSchema)

    def invoke(self, messages: list[Message] | str) -> str:
        """Generate a response using LangChain's ChatOpenAI client."""
        try:
            # Convert Pydantic Message objects to dicts as expected by ChatOpenAI

            if isinstance(messages, str):
                # this is beacuse Is SO TEDIOUS to debug in the console
                # with using str, there is no memory of the previous messages
                messages = [Message(role="user", content=messages)]

            messages_openai_format = [m.model_dump(include={"role", "content"}) for m in messages]

            # TODO
            # Sometimes is failing because don't allow response_format
            # How can I handle that? IDK
            response = self._llm_with_tools.invoke(messages_openai_format)

            # TODO add tests later this complex logic
            if isinstance(response, AIMessage) and len(response.tool_calls) > 0:  # pragma: no cover

                # I don't know when this changed, but the response.content was empty
                # When there is a tool call
                pre_tool_call_content: str = ""
                if response.content != "":
                    pre_tool_call_content += str(response.content)  # type: ignore[misc]
                    for tool_call in response.tool_calls:
                        pre_tool_call_content += (
                            f"\n Tool calls: {tool_call['name']} with args: {tool_call['args']}\n"
                        )

                    pre_tool_call_content += "\n"

                tool_call_message: ToolCallMessageOpenAI = ToolCallMessage(
                    **convert_to_openai_messages(response)  # type: ignore
                ).model_dump()
                tool_call_responses: list[ToolResponseMessageOpenAI] = []
                for tool_call in response.tool_calls:
                    # Call the tool with the arguments provided in the tool call
                    expected_desired_tool = list(
                        filter(lambda row: row.name == tool_call["name"], self._tools)
                    )
                    assert len(expected_desired_tool) == 1, (
                        f"Tool calls with name {tool_call['name']}"
                        "have multiple tools with the "
                        "same name or no tool with this name found"
                    )

                    desired_tool = expected_desired_tool[0]

                    tool_call_response = cast(
                        ToolResponseMessageOpenAI,
                        convert_to_openai_messages(
                            desired_tool.invoke(tool_call)  # type: ignore
                        ),
                    )

                    # Just to check in runtime
                    ToolResponseMessage(**tool_call_response)

                    tool_call_responses.append(tool_call_response)

                    final_message = (
                        messages_openai_format + [tool_call_message] + tool_call_responses
                    )
                    response_with_tool_call = self._llm_with_tools.invoke(
                        final_message  # type: ignore
                    )
                    content = response_with_tool_call.content  # type: ignore
                    assert isinstance(content, str)
                    return pre_tool_call_content + content

            # Handle the content type properly - LangChain response content can be str
            # or complex content
            # Use type: ignore to suppress the "partially unknown" type warning
            content = response.content  # type: ignore[misc]

            # TODO, add a logging here to warn that the return is not a simple string
            assert isinstance(content, str), (
                f"{type(content)} is not a string"  # type: ignore[misc]
                ", but a complex content type. "
                "Please handle this case properly."
            )
            return content

        except Exception as e:
            return f"An error occurred: {e}"


# TODO: Add this main with tool calling and check that don't call wikipedia or other not
# explicitly called tools
# TODO: is calling tools in the second message
if __name__ == "__main__":  # pragma: no cover, JUST DO WHEN RUNNING THIS FILE DIRECTLY()
    from rag_system.use_cases.tools.documents_retrival.retriever_factory import (
        load_documents_tool,
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
    )

    offline_client = OfflineLLMClient(llm, [load_documents_tool])
    print(offline_client.invoke([Message(role="user", content="What is the capital of France?")]))

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
    client = LLMClient(_base_llm=llm, _tools=[load_documents_tool])
    test_messages = [Message(role="user", content="What is the capital of France?")]
    response = client.invoke(test_messages)
    print(response)

    # test that here the model should call the tool
    test_messages = [Message(role="user", content="Give me some documents about France.")]
    response = client.invoke(test_messages)
    print(response)
