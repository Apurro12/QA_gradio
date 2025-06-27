from typing import TypedDict


class EmptyResponse(TypedDict):
    content: None


class Document(TypedDict):
    content: str
    questions: list[str]


Documents = list[Document]


example_docs: Documents = [
    {
        "content": "Our refund policy allows returns within 30 days.",
        "questions": [
            "How can I get a refund?",
            "What is the refund policy?",
            "Can I return a product?",
        ],
    },
    {
        "content": "Shipping takes 3-5 business days.",
        "questions": ["How long does shipping take?", "When will my package arrive?"],
    },
]
