import gradio as gr
from use_cases.agent import Agent

# This Agent class should be modified to an abstract class
def launch_gradio(agent: Agent):
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
    from use_cases.classify_question import QuestionClassifier
    from use_cases.answer_question import AnswerGenerator
    from infrastructure.llm_client import LLMClient
    # This import is being used twice because later I will move
    # The Agent class to a abstract class
    from use_cases.agent import Agent

    llm = LLMClient()
    question_classifier = QuestionClassifier()
    answer_generator = AnswerGenerator(llm)
    agent = Agent(llm, question_classifier, answer_generator)
    launch_gradio(agent)