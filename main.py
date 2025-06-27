import argparse

from rag_system.infrastructure.conversation_manager import InMemoryConversationManager
from rag_system.infrastructure.document_loader import OfflineDocumentLoader
from rag_system.infrastructure.llm_client import LLMClient, OfflineLLMClient
from rag_system.interfaces.gradio_ui import launch_gradio
from rag_system.interfaces.strategies_classifier.exact_match_classifier import ExactMatchClassifier
from rag_system.interfaces.strategies_classifier.llm_classifier import LLMClassifier
from rag_system.use_cases.agent import Agent
from langchain_openai import ChatOpenAI
from rag_system.use_cases.tools.document_loader import load_documents_tool


def create_agent(offline: bool, use_conversation_manager: bool):
    """Create and return configured agent"""

    llm = LLMClient(
        ChatOpenAI(),
        [load_documents_tool]
    )
    
    conversation_manager = InMemoryConversationManager()

    return Agent(llm_client=llm, conversation_manager=conversation_manager)


def main(offline: bool, use_conversation_manager: bool):
    agent = create_agent(offline, use_conversation_manager)
    launch_gradio(agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Run in offline mode")
    parser.add_argument("--conversation-manager", action="store_true", help="Enable conversation manager")
    args = parser.parse_args()
    main(args.offline, args.conversation_manager)
