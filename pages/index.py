import gradio as gr
import os
from openai import OpenAI


def bot_fn(message, history):
    messages = history + [{"role": "user", "content": message}]

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, stream=True
    )

    bot_message = ""
    for _resp in resp:
        if _resp.choices[0].delta.content:
            bot_message += _resp.choices[0].delta.content
            yield bot_message


demo = gr.ChatInterface(bot_fn, type="messages")

if __name__ == "__main__":
    demo.launch()
