from rag_system.domain.agent import BaseAgent
from rag_system.domain.conversation_manager import BaseConversationManager, Message
from rag_system.domain.llm_client import BaseLLMClient
from rag_system.use_cases.llm_client import LLMClient

from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes

class Agent(BaseAgent):
    def __init__(
        self,
        llm_client: BaseLLMClient,
        conversation_manager: BaseConversationManager | None = None,
        tracer_provider: trace.TracerProvider | None = None, 
    ):
        """Initialize the agent with a question classifier.

        answer generator, and optional conversation manager.
        """
        self.llm_client = llm_client
        self.conversation_manager = conversation_manager
        self.tracer_provider = tracer_provider

        # How this work with tracer_provider being None?
        self.tracer = trace.get_tracer(
            "use_cases.agent.Agent", 
            tracer_provider=tracer_provider
        )

    # TODO: Check that is extracting the documents correctly
    def chat(self, message: str, history: list[Message], session_id: str | None = None) -> str:
        """Handle a chat message with conversation context.

        If conversation manager is not provided, fallback to simple question
        answering.
        If there is not a valid documents/context for the question,
        do not add it to the prompt.
        History is not currently used, it is here to comply with gradio interface.
        Maybe it should be used to maintain conversation history in the future.
        """
        with self.tracer.start_as_current_span("agent_chat") as agent_span:
            response = self.llm_client.invoke(history + [Message(role="user", content=message)])

            # TODO: In the future this should save the conversation history in the
            # conversation manager

            if session_id is not None:
                agent_span.set_attribute(SpanAttributes.SESSION_ID, session_id)
            agent_span.set_attribute(SpanAttributes.INPUT_VALUE, message)
            agent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, response) #type: ignore

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

    from rag_system.infrastructure.document_retriever_loader import DOCUMENT_RETRIVAL_CLASS_MAP
    from rag_system.use_cases.tools.documents_retrival.retriever_factory import (
        RETRIEVAL_STRATEGY_CLASS_MAP,
        RETRIEVAL_STRATEGY_WKARGS_MAP,  # type: ignore
        factory_documents_retrieval_tool # type: ignore
    )

    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..","..")
    sys.path.insert(0, os.path.abspath(project_root))

    from config.config import RAGConfig, load_config

    config: RAGConfig = load_config(f"config/default.json")

    ConnectionManager = DOCUMENT_RETRIVAL_CLASS_MAP[config.retrieval.ConnectionManager]
    QueryService = RETRIEVAL_STRATEGY_CLASS_MAP[config.retrieval.QueryService]
    QueryServiceWargs = RETRIEVAL_STRATEGY_WKARGS_MAP(config.retrieval.QueryServiceWargs)

    load_documents_tool = factory_documents_retrieval_tool( #type: ignore
        retrieval_strategy=QueryService,
        document_retriever=ConnectionManager(),
        **QueryServiceWargs # type: ignore
    )

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
