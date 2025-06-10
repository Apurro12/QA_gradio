import argparse
from rag_system.infrastructure.document_loader import OfflineDocumentLoader
from rag_system.infrastructure.llm_client import LLMClient, OfflineLLMClient
from rag_system.interfaces.gradio_ui import launch_gradio
from rag_system.interfaces.strategies_classifier.classify_question import ExactMatchClassifier
from rag_system.use_cases.answer_question import AnswerGenerator, OfflineAnswerGenerator
from rag_system.use_cases.agent import Agent


def main(offline: bool):
    # Wiring: instantiate concrete implementations
    llm = OfflineLLMClient() if offline else LLMClient()
    answer_generator = OfflineAnswerGenerator(llm) if offline else AnswerGenerator(llm)

    # This need to be updated when I have another document loader (e.g. from a vector store)
    offline_document_loader = OfflineDocumentLoader() if offline else OfflineDocumentLoader()
    question_classifier = ExactMatchClassifier(document_loader=offline_document_loader)

    agent = Agent(question_classifier, answer_generator)
    launch_gradio(agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Run in offline mode")
    args = parser.parse_args()
    main(args.offline)
