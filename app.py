# app.py : Multimodal Chatbot
import os
import time
import logging
import jinja2
import pprint
import gradio as gr
from dotenv import load_dotenv
from utils.message import parse_message, render_message, _rerender_message, _rerender_history
from utils.gradio import reload_javascript
from utils import llms

################################################################
# Load environment variables and configure logging
################################################################
load_dotenv()  # Load environment variables from .env file

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)-15s] %(message)s',
    datefmt="%m/%d/%Y %I:%M:%S %p %Z"
)

# Overriding print function to use logging
def print(*args, **kwargs):
    sep = kwargs.get('sep', ' ')
    logger.warning(sep.join(str(arg) for arg in args))

llms.print = print
llms.parse_endpoints_from_environ()

################################################################
# Globals
################################################################
_default_session_state = {
    'context_switch_at': 0,  # History before this point should be ignored (e.g., after uploading an image or file)
    'messages': [],
}

TITLE = "Gradio Multimodal Chatbot Template"
DESCRIPTION = "Welcome"

SETTINGS = {
    'Info': {
        '__metadata': {'open': False, 'tabbed': False},
        'session_state': {'cls': 'State', 'value': _default_session_state},
        'status': {'cls': 'JSON', 'label': 'Status'},
        'show_status_btn': {'cls': 'Button', 'value': 'Show'}
    },
    'Settings': {
        '__metadata': {'open': False, 'tabbed': False},
        'system_prompt': {'cls': 'Textbox', 'lines': 5, 'label': "System Prompt"},
        'speech_synthesis': {'cls': 'Checkbox', 'value': False, 'label': "Speech Synthesis"}
    },
    'Parameters': {
        '__metadata': {'open': True, 'tabbed': False},
        'bot_fn': {
            'cls': 'Dropdown',
            'choices': ['auto', 'random', 'llm'],
            'value': 'auto',
            'label': 'Bot Function',
        },
        'chat_engine': {
            'cls': 'Dropdown', 
            'choices': ['auto', 'gpt-4o-mini', 'gpt-4o'] + list(llms.LLM_ENDPOINTS.keys()), 
            'value': 'auto',
            'label': "Chat Engine"
        },
        'temperature': {'cls': 'Slider', 'minimum': 0, 'maximum': 1, 'value': 0.0, 'step': 0.1, 'label': "Temperature"}
    }
}

COMPONENTS = {}
EXCLUDED_KEYS = ['status', 'show_status_btn']

################################################################
# Utility Functions
################################################################
def _create_from_dict(PARAMS, tabbed=False):
    params = {}
    for name, kwargs in PARAMS.items():
        if name.startswith('__'):
            continue
        cls_ = kwargs.pop('cls')
        if tabbed:
            tab_name = kwargs.get('label', name)
            with gr.Tab(tab_name):
                params[name] = getattr(gr, cls_)(**kwargs)
        else:
            params[name] = getattr(gr, cls_)(**kwargs)
    return params

def _clear(session_state):
    """Clear all session state keys except session_hash."""
    import copy
    session_hash = session_state.get('session_hash')
    session_state.clear()
    session_state.update(copy.deepcopy(_default_session_state))
    if session_hash:
        session_state['session_hash'] = session_hash
    return session_state

def transcribe(audio=None):
    try:
        from utils.azure_speech import speech_recognition
        return speech_recognition(audio)
    except Exception as e:
        return f"Microphone is not supported: {e}"

def _speech_synthesis(text):
    try:
        from utils.azure_speech import speech_synthesis
        speech_synthesis(text=text)
    except Exception as e:
        print(f"Speaker is not supported: {e}")

def _collect_kwargs(SETTINGS, EXCLUDED_KEYS):
    kwargs = {}
    for section, items in SETTINGS.items():
        for k, v in items.items():
            if k not in EXCLUDED_KEYS and not k.startswith('__'):
                kwargs[k] = v.get('value', None)
    return kwargs

def _show_status(*args):
    additional_keys = list(_collect_kwargs(SETTINGS, EXCLUDED_KEYS).keys())
    kwargs = {k: v for k, v in zip(additional_keys, args)}
    return kwargs

def _calc_speed(session_state):
    if 'usage' in session_state and 'elapsed_time' in session_state:
        session_state['speed'] = session_state['usage']['total_tokens'] / session_state['elapsed_time']
        session_state['speed_completion'] = session_state['usage']['completion_tokens'] / session_state['elapsed_time']

################################################################
# Bot Functions
################################################################
from utils.llms import _llm_call, _llm_call_stream, _random_bot_fn

def _slash_bot_fn(message, history, **kwargs):
    cmd, *rest = message[1:].split(' ', maxsplit=1)
    return message

def bot_fn(message, history, **kwargs):
    # Default "auto" behavior
    AUTOS = {'bot_fn': 'llm', 'chat_engine': 'gpt-4o-mini'}
    for param, default_value in AUTOS.items():
        kwargs[param] = default_value if kwargs[param] == 'auto' else kwargs[param]

    if not history or message == '/clear':
        _clear(kwargs['session_state'])

    if message.startswith('/') or message.startswith('.'):
        bot_message = _slash_bot_fn(message, history, **kwargs)
    else:
        bot_fn_map = {
            'random': _random_bot_fn,
            'llm': _llm_call_stream,
        }
        bot_message = bot_fn_map[kwargs['bot_fn']](message, history, **kwargs)

    if isinstance(bot_message, str):
        yield bot_message
    else:
        yield from bot_message
    return bot_message

def bot_fn_wrapper(message, history, request: gr.routes.Request, *args):
    kwargs = _collect_kwargs(SETTINGS, EXCLUDED_KEYS)
    kwargs = {k: v for k, v in zip(kwargs.keys(), args)}
    if request:
        kwargs['session_state']['session_hash'] = request.session_hash

    __TIC = time.time()
    bot_message = yield from bot_fn(message, history, **kwargs)

    if kwargs.get('speech_synthesis', False):
        _speech_synthesis(_rerender_message(bot_message, format='speech'))

    kwargs['session_state']['elapsed_time'] = time.time() - __TIC
    _calc_speed(kwargs['session_state'])
    print(pprint.pformat(kwargs))
    return bot_message

def bot_fn_wrapper_prod(message, history, request: gr.routes.Request, session_state):
    kwargs = _collect_kwargs(SETTINGS, EXCLUDED_KEYS)
    kwargs['session_state'] = session_state
    if request:
        kwargs['session_state']['session_hash'] = request.session_hash

    __TIC = time.time()
    bot_message = yield from bot_fn(message, history, **kwargs)

    if kwargs.get('speech_synthesis', False):
        _speech_synthesis(_rerender_message(bot_message, format='speech'))

    kwargs['session_state']['elapsed_time'] = time.time() - __TIC
    _calc_speed(kwargs['session_state'])
    print(pprint.pformat(kwargs))
    return bot_message

################################################################
# Gradio Demo Setup
################################################################
def get_demo():
    global COMPONENTS
    css = """
    #chatbot {
        min-height: 600px;
    }
    .full-container label {
        display: block;
        padding-left: 8px;
        padding-right: 8px;
    }
    """
    with gr.Blocks(css=css) as demo:
        gr.HTML(f"<center><h1>{TITLE}</h1></center>")
        with gr.Accordion("Expand for Introduction and Usage", open=False):
            gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                for section_name, settings in SETTINGS.items():
                    metadata = settings['__metadata']
                    with gr.Accordion(section_name, open=metadata.get('open', False)):
                        COMPONENTS.update(_create_from_dict(settings, tabbed=metadata.get('tabbed', False)))

            additional_keys = list(_collect_kwargs(SETTINGS, EXCLUDED_KEYS).keys())
            additional_inputs=[COMPONENTS[key] for key in additional_keys]
            with gr.Column(scale=9):
                from utils.utils import change_signature
                sig_bot_fn = change_signature(['message', 'history', 'request'] + additional_keys)(bot_fn_wrapper)
                from utils.gradio import ChatInterface
                chatbot = ChatInterface(
                    sig_bot_fn, 
                    type='messages', 
                    additional_inputs=additional_inputs,
                    multimodal=False,
                    avatar_images=('assets/user.png', 'assets/bot.png')
                )
                chatbot.audio_btn.click(transcribe, [], [chatbot.textbox], queue=False)
                if 'status' in COMPONENTS and 'show_status_btn' in COMPONENTS:
                    COMPONENTS['show_status_btn'].click(_show_status, additional_inputs, COMPONENTS['status'], queue=False)

                with gr.Accordion("Examples", open=False):
                    gr.Examples(
                        ["What's the Everett interpretation of quantum mechanics?"],
                        inputs=chatbot.textbox
                    )

    reload_javascript()
    return demo

def get_demo_prod():
    from utils.gradio import ChatInterfaceProd
    css = """
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
    with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:
        session_state = gr.State(_default_session_state)
        chatbot = ChatInterfaceProd(
            bot_fn_wrapper_prod, 
            type='messages', 
            additional_inputs=[session_state],
            avatar_images=('assets/user.png', 'assets/bot.png'),
        )
    reload_javascript()
    return demo

def main(args):
    if args.env in ['dev', 'prod']:
        demo = get_demo_prod() if args.env == 'prod' else get_demo()
        demo.queue().launch(server_name='0.0.0.0', server_port=args.port)

    elif args.env == 'prod_fastapi':
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
        import gradio as gr
        import uvicorn

        app = FastAPI()

        @app.get("/", response_class=HTMLResponse)
        def index():
            with open('assets/webchat.gradio.html') as f:
                index_html = f.read().replace("http://localhost:7860", f"http://localhost:{args.port}{args.mount_path}")
            return index_html

        demo = get_demo_prod()
        app = gr.mount_gradio_app(app, demo, path=args.mount_path)

        uvicorn.run(app, port=args.port)

    else:
        raise ValueError(f"Invalid environment: {args.env}")

def parse_args():
    """Parse input arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Multimodal Chatbot',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-p', '--port', default=7860, type=int,
        help='port number.')
    parser.add_argument(
        '--debug', action='store_true', 
        help='debug mode.')
    parser.add_argument(
        '--env', type=str, default='dev', choices=['dev', 'prod', 'prod_fastapi'], 
        help='Environment.')
    parser.add_argument(
        '--mount-path', type=str, default='/demo', 
        help='Mount path for gradio app.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

