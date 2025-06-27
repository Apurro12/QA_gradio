from rag_system.domain.agent import BaseAgent
from rag_system.domain.answer_generator import BaseAnswerGenerator
from rag_system.domain.conversation_manager import BaseConversationManager, Message
from rag_system.interfaces.strategies_classifier.exact_match_classifier import (
    BaseQuestionClassifier,
)


class Agent(BaseAgent):
    def __init__(
        self,
        question_classifier: BaseQuestionClassifier,
        answer_generator: BaseAnswerGenerator,
        conversation_manager: BaseConversationManager | None = None,
    ):
        """Initialize the agent with a question classifier.

        answer generator, and optional conversation manager.
        """
        self.question_classifier = question_classifier
        self.answer_generator = answer_generator
        self.conversation_manager = conversation_manager

    # TODO: Check that is extracting the documents correctly
    def chat(self, message: str, history: list[Message]) -> str:
        """Handle a chat message with conversation context.

        If conversation manager is not provided, fallback to simple question
        answering.
        If there is not a valid documents/context for the question,
        do not add it to the prompt.
        History is not currently used, it is here to comply with gradio interface.
        Maybe it should be used to maintain conversation history in the future.
        """
        response = self.answer_generator.generate(
            history + [Message(role="user", content=message)]
        )

        # TODO: In the future this should save the conversation history in the
        # conversation manager
        if self.conversation_manager:
            self.conversation_manager.update_conversation(
                history
                + [
                    Message(role="user", content=message),
                    Message(role="assistant", content=response),
                ]
            )

        return response


if __name__ == "__main__":  # pragma: no cover, JUST DO WHEN RUNNING THIS FILE DIRECTLY
    from rag_system.infrastructure.conversation_manager import (
        InMemoryConversationManager,
    )
    from rag_system.infrastructure.document_loader import OfflineDocumentLoader
    from rag_system.infrastructure.llm_client import OfflineLLMClient
    from rag_system.interfaces.strategies_classifier.exact_match_classifier import (
        ExactMatchClassifier,
    )
    from rag_system.use_cases.answer_generator import OfflineAnswerGenerator

    llm = OfflineLLMClient()
    offline_document_loader = OfflineDocumentLoader()
    question_classifier = ExactMatchClassifier(document_loader=offline_document_loader)
    answer_generator = OfflineAnswerGenerator(llm=llm)
    conversation_manager = InMemoryConversationManager()

    agent = Agent(question_classifier, answer_generator, conversation_manager)

    # Interactive chat example
    print("Chat interface started. Type 'quit' to exit, 'clear' to clear conversation.")
    history: list[Message] = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        history.append(Message(role="user", content=user_input))
        response = agent.chat(user_input, history)
        history.append(Message(role="assistant", content=response))

        print(f"Assistant: {response}")
        print()
        print("----------------")
        print("----------------")
        print("----------------")
        print()
