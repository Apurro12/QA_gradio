
from rag_system.infrastructure.llm_client import LLMClient
from rag_system.interfaces.gradio_ui import launch_gradio
from rag_system.use_cases.classify_question import QuestionClassifier
from rag_system.use_cases.answer_question import AnswerGenerator
from rag_system.use_cases.agent import Agent
from rag_system.domain.document import example_docs


def main():
    # Wiring: instantiate concrete implementations
    llm = LLMClient()
    question_classifier = QuestionClassifier(documents = example_docs)
    answer_generator = AnswerGenerator(llm)

    # Create the agent with injected dependencies
    agent = Agent(llm, question_classifier, answer_generator)

    # Launch the Gradio interface
    launch_gradio(agent)

if __name__ == "__main__":
    main()
