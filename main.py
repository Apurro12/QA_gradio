from interfaces.gradio_ui import launch_gradio
from infrastructure.llm_client import LLMClient
from use_cases.classify_question import QuestionClassifier
from use_cases.answer_question import AnswerGenerator
from use_cases.agent import Agent
from tests.example_docs import example_docs

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
