import argparse
from rag_system.infrastructure.document_loader import OfflineDocumentLoader
from rag_system.infrastructure.llm_client import LLMClient, OfflineLLMClient
from rag_system.interfaces.gradio_ui import launch_gradio
from rag_system.interfaces.strategies_classifier.exact_match_classifier import ExactMatchClassifier
from rag_system.interfaces.strategies_classifier.llm_classifier import LLMClassifier
from rag_system.use_cases.answer_generator import AnswerGenerator, OfflineAnswerGenerator
from rag_system.use_cases.agent import Agent
from rag_system.infrastructure.conversation_manager import InMemoryConversationManager


def main(offline: bool, use_conversation_manager: bool):
    # Wiring: instantiate concrete implementations
    llm = OfflineLLMClient() if offline else LLMClient()
    answer_generator = OfflineAnswerGenerator(llm) if offline else AnswerGenerator(llm)

    # This need to be updated when I have another document loader (e.g. from a vector store)
    offline_document_loader = OfflineDocumentLoader() if offline else OfflineDocumentLoader()
    question_classifier = ExactMatchClassifier(document_loader=offline_document_loader) if offline else LLMClassifier(offline_document_loader, llm)

    # Conditionally add conversation manager
    conversation_manager = InMemoryConversationManager() if use_conversation_manager else None

    agent = Agent(question_classifier, answer_generator, conversation_manager)
    launch_gradio(agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Run in offline mode")
    parser.add_argument("--conversation-manager", action="store_true", help="Enable conversation manager")
    args = parser.parse_args()
    main(args.offline, args.conversation_manager)
