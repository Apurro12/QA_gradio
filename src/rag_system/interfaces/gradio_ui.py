import gradio as gr
from rag_system.domain.agent import BaseAgent


# This Agent class should be modified to an abstract class
def launch_gradio(agent: BaseAgent):
    def respond_to_question(user_question: str) -> str:
        return agent.respond_user_question(user_question)

    iface = gr.Interface(
        fn=respond_to_question,
        inputs=gr.Textbox(lines=2, placeholder="Ask your question here..."),
        outputs="text",
        title="RAG Agent Q&A",
        description="Ask questions and get answers based on documents."
    )

    iface.launch()

if __name__ == "__main__":
    from rag_system.use_cases.classify_question import QuestionClassifier
    from rag_system.use_cases.answer_question import AnswerGenerator
    from rag_system.infrastructure.llm_client import LLMClient
    from rag_system.domain.document import example_docs
    from rag_system.use_cases.agent import Agent
    

    # Wiring: instantiate concrete implementations
    llm = LLMClient()
    question_classifier = QuestionClassifier(documents = example_docs)
    answer_generator = AnswerGenerator(llm)

    # Create the agent with injected dependencies
    agent = Agent(llm, question_classifier, answer_generator)

    # Launch the Gradio interface
    launch_gradio(agent)