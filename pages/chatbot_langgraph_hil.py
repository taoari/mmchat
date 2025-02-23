import gradio as gr
import os
import re
import io
from PIL import Image
from openai import OpenAI
from config.config import LLM_ENDPOINTS

from langgraph.types import Command, interrupt
from graphs.chatbot_hil import graph, get_interrupts
from utils.deepseek import format_deepseek_message


def get_messages(message, history, system_prompt=None):
    messages = history + [{"role": "user", "content": message}]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}]
    return messages


def bot_fn(message, history, request: gr.routes.Request, chat_model):
    config = {"configurable": {"thread_id": request.session_hash}}

    snapshot = graph.get_state(config)
    interrupts = get_interrupts(snapshot)

    if interrupts:
        human_command = Command(resume={"data": message})
        response = graph.invoke(human_command, config)
    else:
        response = graph.invoke(
            {"messages": [{"role": "user", "content": message}]}, config
        )

    yield format_deepseek_message(response["messages"][-1].content), response

    snapshot = graph.get_state(config)
    interrupts = get_interrupts(snapshot)

    if interrupts:
        yield f"Prompt: {interrupts}", response

    if request:
        print(f"Request headers dictionary: {request.headers}")
        print(f"Session hash: {request.session_hash}")


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
