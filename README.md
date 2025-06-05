## to run standalone files go to the parent folder and run 
uv run python -m domain.agent  
uv run python -m infrastructure.llm_client

# to lunch gradio
uv run python -m interfaces.gradio_ui

# To DO:
* Add abstract class to the vector store (I'm still not using it)
* Modify the documents to query a db or similar
* Add logging, should I log into grafana/langsmith/phoenix/console?