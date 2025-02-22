import gradio as gr
import os
import re

from openai import OpenAI
from config import config, secrets
from config.config import LLM_ENDPOINTS
from utils.deepseek import format_deepseek_message


def get_messages(message, history, system_prompt=None):
    messages = history + [{"role": "user", "content": message}]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}]
    return messages


def bot_fn(message, history, system_prompt, chat_model):
    messages = get_messages(message, history, system_prompt)

    assert chat_model in LLM_ENDPOINTS

    model_name = LLM_ENDPOINTS[chat_model]["model_name"]

    base_url = LLM_ENDPOINTS[chat_model]["base_url"]
    provider = LLM_ENDPOINTS[chat_model]["provider"]
    api_key = {
        "onprem": "-",
        "openai": secrets.OPENAI_API_KEY,
        "openrouter": secrets.OPENROUTER_API_KEY,
    }[provider]

    client = OpenAI(api_key=api_key, base_url=base_url)

    # resp = client.chat.completions.create(
    #     model=model_name, messages=messages, stream=True
    # )

    # bot_message = ""
    # for _resp in resp:
    #     if _resp.choices[0].delta.content:
    #         bot_message += _resp.choices[0].delta.content
    #         yield format_deepseek_message(bot_message), {}
    # print(bot_message)

    resp = client.chat.completions.create(
        model=model_name, messages=messages, stream=False
    )
    bot_message = resp.choices[0].message.content
    usage = resp.usage.__dict__
    info = dict(messages=messages, usage=usage)
    return format_deepseek_message(bot_message), info


def get_demo():
    with gr.Blocks() as demo:
        with gr.Tab("Settings"):
            system_prompt = gr.Textbox(lines=5, label="System prompt")
            chat_model = gr.Dropdown(
                choices=list(LLM_ENDPOINTS.keys()),
                label="Chat model",
            )
        with gr.Tab("State"):
            state = gr.JSON(label="State")

        chat_interface = gr.ChatInterface(
            bot_fn,
            type="messages",
            additional_inputs=[system_prompt, chat_model],
            additional_outputs=[state],
        )
    return demo


demo = get_demo()

if __name__ == "__main__":
    demo.launch()
