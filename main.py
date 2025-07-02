import argparse

from langchain_openai import ChatOpenAI
from config.config import load_config
from config.config import RAGConfig
from rag_system.infrastructure.conversation_manager import InMemoryConversationManager
from rag_system.infrastructure.document_retriever_loader import DOCUMENT_RETRIVAL_CLASS_MAP
from rag_system.interfaces.gradio_ui import launch_gradio
from rag_system.use_cases.agent import Agent
from rag_system.use_cases.llm_client_factory import LLM_CLIENT_CLASS_MAP
from rag_system.use_cases.tools.documents_retrival.retriever_factory import (
    RETRIEVAL_STRATEGY_CLASS_MAP,
    RETRIEVAL_STRATEGY_WKARGS_MAP,  # type: ignore
    factory_documents_retrieval_tool # type: ignore
)

def create_agent(config: RAGConfig) -> Agent:
    """Create and return configured agent."""

    
    ConnectionManager = DOCUMENT_RETRIVAL_CLASS_MAP[config.retrieval.ConnectionManager]
    QueryService = RETRIEVAL_STRATEGY_CLASS_MAP[config.retrieval.QueryService]
    QueryServiceWargs = RETRIEVAL_STRATEGY_WKARGS_MAP(config.retrieval.QueryServiceWargs)

    load_documents_tool = factory_documents_retrieval_tool( #type: ignore
        retrieval_strategy=QueryService,
        document_retriever=ConnectionManager(),
        **QueryServiceWargs # type: ignore
    )

    LLM_CLIENT = LLM_CLIENT_CLASS_MAP[config.llm.mode]
    llm = LLM_CLIENT(ChatOpenAI(), [load_documents_tool])

    conversation_manager = InMemoryConversationManager()

    return Agent(llm, conversation_manager)

def main(config_name: str | None = None, offline: bool = False):
    """Create agent and launch Gradio UI."""
    if offline:
        config_name = "offline"

    # If offline mode is on, override the configuration name
    config: RAGConfig = load_config(f"config/{config_name}.json" if config_name else "config/default.json")
    
    # Get agent and launch UI
    agent = create_agent(config)
    launch_gradio(agent, 
                 host=config.ui.host,
                 port=config.ui.port,
                 share=config.ui.share)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RAG System with configurable settings")
    parser.add_argument("--config-name", type=str, help="Name of predefined configuration (e.g., 'offline', 'kubernetes')")
    parser.add_argument("--offline", action="store_true", help="Override the configuration to use offline mode")
    
    args = parser.parse_args()

    main(config_name=args.config_name, offline=args.offline)
