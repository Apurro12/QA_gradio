from rag_system.domain.agent import BaseAgent
from rag_system.domain.conversation_manager import BaseConversationManager, Message
from rag_system.use_cases.llm_client import LLMClient
from rag_system.domain.llm_client import BaseLLMClient


class Agent(BaseAgent):
    def __init__(
        self,
        llm_client: BaseLLMClient,
        conversation_manager: BaseConversationManager | None = None,
    ):
        """Initialize the agent with a question classifier.

        answer generator, and optional conversation manager.
        """
        self.llm_client = llm_client
        self.conversation_manager = conversation_manager

    # TODO: Check that is extracting the documents correctly
    def chat(self, message: str, history: list[Message]) -> str:
        """Handle a chat message with conversation context.

        If conversation manager is not provided, fallback to simple question
        answering.
        If there is not a valid documents/context for the question,
        do not add it to the prompt.
        History is not currently used, it is here to comply with gradio interface.
        Maybe it should be used to maintain conversation history in the future.
        """
        response = self.llm_client.invoke(
            history + [Message(role="user", content=message)]
        )

        # TODO: In the future this should save the conversation history in the
        # conversation manager
        if self.conversation_manager:
            self.conversation_manager.update_conversation(
                history
                + [
                    Message(role="user", content=message),
                    Message(role="assistant", content=response),
                ]
            )

        return response


if __name__ == "__main__":  # pragma: no cover, JUST DO WHEN RUNNING THIS FILE DIRECTLY
    from langchain_openai import ChatOpenAI

    from rag_system.infrastructure.conversation_manager import (
        InMemoryConversationManager,
    )
    from rag_system.use_cases.tools.document_loader import load_documents_tool

    llm = LLMClient(ChatOpenAI(), [load_documents_tool])

    conversation_manager = InMemoryConversationManager()

    agent = Agent(llm_client=llm, conversation_manager=conversation_manager)

    # Interactive chat example
    print("Chat interface started. Type 'quit' to exit.")
    history: list[Message] = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        response = agent.chat(user_input, history)
        history.append(Message(role="user", content=user_input))
        history.append(Message(role="assistant", content=response))

        print(f"Assistant: {response}")
        print()
        print("----------------")
        print("----------------")
        print("----------------")
        print()
