from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import gradio as gr

app = FastAPI()

css="""
#chatbot {
    min-height: 600px;
}
footer {
    display: none !important;
}
.message-row {
    margin: 8px 5px 2px 5px;
}
.full-container label {
    display: block;
    padding: 0 8px;
}
    """

@app.get("/", response_class=HTMLResponse)
def index():
    with open('webchat.gradio.html') as f:
        index_html = f.read().replace("http://localhost:7860", "http://localhost:7860/demo")
    return index_html

def get_demo():
    from utils.llms import _llm_call, _llm_call_stream, _random_bot_fn
    from utils.gradio import ChatInterfaceProd, reload_javascript

    def bot_fn(message, history):
        # bot_message = _llm_call_stream(message, history)
        # yield from bot_message
        bot_message = _random_bot_fn(message, history)
        return bot_message

    with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:
        chatbot = ChatInterfaceProd(bot_fn, type='messages',
                avatar_images=('assets/user.png', 'assets/bot.png'))
    reload_javascript()
    return demo

demo = get_demo()
app = gr.mount_gradio_app(app, demo, path='/demo')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=7860)