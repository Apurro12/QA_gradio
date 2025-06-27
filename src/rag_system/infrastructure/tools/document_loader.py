from langchain_core.tools import tool # type: ignore
from typing import Annotated


@tool
def load_documents(user_input: Annotated[str, "The description of the documents the user wants to retrieve"]) -> str:
   """
      Load documents from a source.
      This function function loads documents based on the user's input.
      It is not necesary that the user explicitely asks for documents,
      but the system can infer that the user wants to retrieve documents based on their input.
      If the question is from common sense or common knowledge do not call this tool.
   """
   return f"Document1: I'm a very important document. I contain information about {user_input}.\n"