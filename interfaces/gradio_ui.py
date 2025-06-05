import gradio as gr
from use_cases.agent import LLM

llm = LLM()

def respond(question):
    return llm.respond(question)

iface = gr.Interface(fn=respond, inputs="text", outputs="text", title="RAG System")
iface.launch()