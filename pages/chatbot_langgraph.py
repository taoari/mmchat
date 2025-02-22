import gradio as gr
import os
import re
import io
from PIL import Image
from openai import OpenAI
from config.config import LLM_ENDPOINTS

# from utils.rag import graph
from graphs.chatbot import graph
from utils.deepseek import format_deepseek_message


def get_messages(message, history, system_prompt=None):
    messages = history + [{"role": "user", "content": message}]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}]
    return messages


def bot_fn(message, history, request: gr.routes.Request, chat_model):
    config = {"configurable": {"thread_id": request.session_hash}}
    response = graph.invoke(
        {"messages": [{"role": "user", "content": message}]}, config
    )

    if request:
        print(f"Request headers dictionary: {request.headers}")
        print(f"Session hash: {request.session_hash}")
    return format_deepseek_message(response["messages"][-1].content), response


def get_demo():
    with gr.Blocks() as demo:
        with gr.Tab("Settings"):
            # system_prompt = gr.Textbox(lines=5, label="System prompt")
            chat_model = gr.Dropdown(
                choices=list(LLM_ENDPOINTS.keys()),
                label="Chat model",
            )
        with gr.Tab("State"):
            state = gr.JSON(open=False, label="State")
        with gr.Tab("Graph"):
            cg = gr.Image(Image.open(io.BytesIO(graph.get_graph().draw_mermaid_png())))
        chat_interface = gr.ChatInterface(
            bot_fn,
            type="messages",
            additional_inputs=[chat_model],
            additional_outputs=[state],
        )
    return demo


demo = get_demo()

if __name__ == "__main__":
    demo.launch()
