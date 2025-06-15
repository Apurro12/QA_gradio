import gradio as gr # pragma: no cover
from rag_system.domain.agent import BaseAgent # pragma: no cover

# NEVER NEVER CHANGE THIS FILE
# THIS FILE IS NOT TESTED

# This file is used to launch the Gradio interface for the RAG agent.
# It is designed to be run as a standalone script.

def launch_gradio(agent: BaseAgent): # pragma: no cover
    def respond_to_question(message: str) -> str:
        return agent.respond_user_question(message)

    iface = gr.ChatInterface(
        fn=respond_to_question,
        title="RAG Agent Chat",
        description="Have a conversation and ask questions based on documents.",
        textbox=gr.Textbox(placeholder="Ask your question here...", container=False, scale=7)
    )

    iface.launch()

if __name__ == "__main__": # pragma: no cover
    from rag_system.interfaces.strategies_classifier.exact_match_classifier import ExactMatchClassifier
    from rag_system.use_cases.answer_generator import OfflineAnswerGenerator
    from rag_system.infrastructure.document_loader import OfflineDocumentLoader
    from rag_system.infrastructure.llm_client import OfflineLLMClient
    from rag_system.use_cases.agent import Agent

    llm = OfflineLLMClient()
    offline_document_loader = OfflineDocumentLoader()
    question_classifier = ExactMatchClassifier(document_loader=offline_document_loader)
    answer_generator = OfflineAnswerGenerator(llm=llm)
    agent = Agent(question_classifier, answer_generator)

    launch_gradio(agent)