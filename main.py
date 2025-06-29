import argparse

from langchain_openai import ChatOpenAI

from rag_system.infrastructure.conversation_manager import InMemoryConversationManager
from rag_system.interfaces.gradio_ui import launch_gradio
from rag_system.use_cases.agent import Agent
from rag_system.use_cases.llm_client import LLMClient, OfflineLLMClient
from rag_system.use_cases.tools.document_loader import load_documents_tool


def create_agent(offline: bool):
    """Create and return configured agent."""
    llm = (
        OfflineLLMClient(ChatOpenAI(), [load_documents_tool])
        if offline
        else LLMClient(ChatOpenAI(), [load_documents_tool])
    )

    conversation_manager = InMemoryConversationManager()

    return Agent(llm, conversation_manager)


def main(offline: bool):
    """Create agent and launch Gradio UI."""
    agent = create_agent(offline)
    launch_gradio(agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Run in offline mode")
    args = parser.parse_args()
    main(args.offline)
