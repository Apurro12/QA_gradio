import gradio as gr  # pragma: no cover

from uuid import uuid4 as uuid  # pragma: no cover
from rag_system.domain.agent import BaseAgent  # pragma: no cover
from rag_system.domain.conversation_manager import (
    GradioHistoryMessage,
    Message,
)  # pragma: no cover

# NEVER NEVER CHANGE THIS FILE
# THIS FILE IS NOT TESTED

# This file is used to launch the Gradio interface for the RAG agent.
# It is designed to be run as a standalone script.

def make_respond_to_question(agent: BaseAgent):
    """Create a function to respond to questions using the RAG agent."""
    def respond_to_question(
            message: str, 
            history: list[GradioHistoryMessage],
            session_id: str | None  # This is used to track the session in Gradio
            ) -> str:
        history_list: list[Message] = list(
            map(
                lambda gradio_message: Message(
                    role=gradio_message["role"], content=gradio_message["content"]
                ),
                history,
            )
        )
        return agent.chat(message, history_list, session_id=session_id)

    return respond_to_question

# This will store the session ID in the state
def generate_session_id(request: gr.Request):
    return str(uuid())

def make_interface(agent: BaseAgent):
    """Create a Gradio interface for the RAG agent."""
    with gr.Blocks() as demo:
        session_state = gr.State()

        chat = gr.ChatInterface(
            fn=make_respond_to_question(agent),
            title="RAG Agent Chat",
            description="Have a conversation and ask questions based on documents.",
            additional_inputs=[session_state],
            textbox=gr.Textbox(placeholder="Ask your question here...", container=False, scale=7),
            type="messages",  # This is the type of the input.
            # It should be an OpenAI style:
            # [{"role": "user"/"assistant", "content": "your question"}]
        )

        demo.load(
            generate_session_id,
            inputs=None,
            outputs=session_state,
            #every=True
        )

    return demo


def launch_gradio(agent: BaseAgent, host: str, port: int, share: bool):  # pragma: no cover
    """Launch the Gradio interface for the RAG agent."""
    iface = make_interface(agent)
    iface.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":  # pragma: no cover
    from langchain_openai import ChatOpenAI

    from rag_system.infrastructure.conversation_manager import (
        InMemoryConversationManager,
    )
    from rag_system.use_cases.agent import Agent
    from rag_system.use_cases.llm_client import OfflineLLMClient
    from rag_system.use_cases.tools.documents_retrival.retriever_factory import (
        load_documents_tool,
    )

    llm = OfflineLLMClient(ChatOpenAI(), [load_documents_tool])
    conversation_manager = InMemoryConversationManager()
    agent = Agent(llm, conversation_manager)

    launch_gradio(agent, host="127.0.0.1", port=7860, share=False)
