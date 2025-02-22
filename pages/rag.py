import gradio as gr
import os
import re
from openai import OpenAI
from config.config import LLM_ENDPOINTS
from utils.rag import graph
from utils.deepseek import format_deepseek_message


def get_messages(message, history, system_prompt=None):
    messages = history + [{"role": "user", "content": message}]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}]
    return messages


def rag_bot_fn(message, history, request: gr.routes.Request, chat_model):
    response = graph.invoke({"question": message, "chat_model": chat_model})

    if request:
        print(f"Request headers dictionary: {request.headers}")
        print(f"Session hash: {request.session_hash}")
    return format_deepseek_message(response["answer"]), response


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
        chat_interface = gr.ChatInterface(
            rag_bot_fn,
            type="messages",
            additional_inputs=[chat_model],
            additional_outputs=[state],
        )
    return demo


demo = get_demo()

if __name__ == "__main__":
    demo.launch()
