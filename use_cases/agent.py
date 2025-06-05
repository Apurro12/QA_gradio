from use_cases.classify_question import BaseQuestionClassifier
from use_cases.answer_question import AnswerGenerator
from infrastructure.llm_client import BaseLLMClient

class Agent:
    def __init__(
            self, 
            llm: BaseLLMClient, 
            question_classifier: BaseQuestionClassifier,
            answer_generator: AnswerGenerator
            ):
        self.llm = llm
        self.question_classifier = question_classifier
        self.answer_generator = answer_generator

    def respond_user_question(self, question: str) -> str:
        """
        Respond to a question by classifying it, selecting a document, and generating an answer.
        """
        document = self.question_classifier.classify(question)
        context = document["content"]

        # This is a placeholder for more sophisticated context retrieval logic
        if not context:
            return "No relevant information found to answer your question."

        response = self.answer_generator.generate(question, context)

        return response

if __name__ == "__main__":
    from use_cases.classify_question import QuestionClassifier
    from use_cases.answer_question import AnswerGenerator
    from infrastructure.llm_client import LLMClient

    llm = LLMClient()
    question_classifier = QuestionClassifier()
    answer_generator = AnswerGenerator(llm)
    
    agent = Agent(llm, question_classifier, answer_generator)
    print(agent.respond_user_question("How can I get a refund?"))