import gradio as gr  # pragma: no cover

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

    def respond_to_question(message: str, history: list[GradioHistoryMessage]) -> str:
        history_list: list[Message] = list(
            map(
                lambda gradio_message: Message(
                    role=gradio_message["role"], content=gradio_message["content"]
                ),
                history,
            )
        )
        return agent.chat(message, history_list)

    return respond_to_question


def make_interface(agent: BaseAgent):
    """Create a Gradio interface for the RAG agent."""
    iface = gr.ChatInterface(
        fn=make_respond_to_question(agent),
        title="RAG Agent Chat",
        description="Have a conversation and ask questions based on documents.",
        textbox=gr.Textbox(
            placeholder="Ask your question here...", container=False, scale=7
        ),
        type="messages",  # This is the type of the input.
        # It should be an OpenAI style:
        # [{"role": "user"/"assistant", "content": "your question"}]
    )

    return iface


def launch_gradio(agent: BaseAgent):  # pragma: no cover
    """Launch the Gradio interface for the RAG agent."""
    iface = make_interface(agent)
    iface.launch()


if __name__ == "__main__":  # pragma: no cover
    from rag_system.infrastructure.document_loader import OfflineDocumentLoader
    from rag_system.interfaces.strategies_classifier.exact_match_classifier import (
        ExactMatchClassifier,
    )
    from rag_system.use_cases.agent import Agent
    from rag_system.use_cases.answer_generator import OfflineAnswerGenerator
    from rag_system.use_cases.llm_client import OfflineLLMClient

    llm = OfflineLLMClient()
    offline_document_loader = OfflineDocumentLoader()
    question_classifier = ExactMatchClassifier(document_loader=offline_document_loader)
    answer_generator = OfflineAnswerGenerator(llm=llm)
    agent = Agent(question_classifier, answer_generator)

    launch_gradio(agent)
