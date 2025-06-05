import gradio as gr
from use_cases.classify_question import classify_question
from use_cases.answer_question import answer_question

def respond(question):
    context = classify_question(question)
    answer = answer_question(question, context)
    return answer

iface = gr.Interface(fn=respond, inputs="text", outputs="text", title="RAG System")
iface.launch()