
from typing import List, TypedDict

class EmptyResponse(TypedDict):
    content: None

class Document(TypedDict):
    content: str
    questions: List[str]

Documents = List[Document]