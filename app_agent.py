# app.py : Multimodal Chatbot
import os
import json
import time
import logging
import jinja2
import pprint
import gradio as gr

from utils.message import parse_message, render_message, _rerender_message, _rerender_history

################################################################
# Load .env and logging
################################################################

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN, format='%(asctime)-15s] %(message)s', datefmt="%m/%d/%Y %I:%M:%S %p %Z")

def print(*args, **kwargs):
    sep = kwargs['sep'] if 'sep' in kwargs else ' '
    logger.warning(sep.join([str(val) for val in args])) # use level WARN for print, as gradio level INFO print unwanted messages

from utils import llms, prompts
llms.print = print

################################################################
# Globals
################################################################

from utils import tools as TOOLS
import tool2schema
TOOLS_SCHEMA = tool2schema.FindToolEnabledSchemas(TOOLS)
print('Tools:\n' + json.dumps(TOOLS_SCHEMA, indent=2))

# DIRECT_RESPONSE_TOOLS = ['get_delivery_date'] # use function output directly instead of LLM rewrite
DIRECT_RESPONSE_TOOLS = [] # always rewrite function output with LLM
AVAILABLE_TOOLS = [obj["function"]["name"] for obj in TOOLS_SCHEMA if obj["type"] == "function"]

_default_session_state = dict(
        messages = [],
        context_switch_at=0, # history before context_switch_at should be ignored (e.g. upload an image or a file)
        message=None,
        previous_message=None,
    )

TITLE = """ Gradio Multimodal Chatbot Template """

DESCRIPTION = """Welcome
"""

SETTINGS = {
    'Info': {
        '__metadata': {'open': False, 'tabbed': False},
        'session_state': dict(cls='State', value=_default_session_state),
        'status': dict(cls='JSON', label='Status'),
        'show_status_btn': dict(cls='Button', value='Show')
    },
    'Settings': {
        '__metadata': {'open': True, 'tabbed': False},
        'system_prompt': dict(cls='Textbox', interactive=True, value=prompts.PROMPT_CHECK_DELIVERY_DATE.strip(), lines=5, label="System prompt"),
        'chat_engine': dict(cls='Dropdown', choices=['auto', 'random', 'gpt-3.5-turbo', 'gpt-4o', 'openai_agent', 'langchain_agent'], value='openai_agent', 
                interactive=True, label="Chat engine"),
        'tools': dict(cls='CheckboxGroup', choices=AVAILABLE_TOOLS, 
                value=AVAILABLE_TOOLS,
                interactive=True, label='Tools'),
        'speech_synthesis': dict(cls='Checkbox', value=False, 
                interactive=True, label="Speech Synthesis"),
    },
    'Parameters': {
        '__metadata': {'open': False, 'tabbed': False},
        'temperature': dict(cls='Slider', minimum=0, maximum=1, value=0.7, step=0.1, interactive=True, label="Temperature")
    }
}

COMPONENTS = {}

COMPONENTS_EXCLUDED = {}
EXCLUDED_KEYS = ['status', 'show_status_btn'] # keys are excluded for chatbot additional inputs

################################################################
# Utils
################################################################

def convert_tools_for_langchain(TOOLS):
    import pdb; pdb.set_trace()
    from langchain_core.tools import Tool, StructuredTool
    
    functions = tool2schema.FindToolEnabled(TOOLS)
    def _convert(func):
        schema = func.to_json()
        # TODO: arguments
        return StructuredTool(name=schema['function']['name'], func=func.func, description=schema['function']['description'], args_schema=schema['function']['parameters'])
    return [_convert(func) for func in functions]

def _create_from_dict(PARAMS, tabbed=False):
    params = {}
    for name, kwargs in PARAMS.items():
        if name.startswith('__'):
            continue
        cls_ = kwargs['cls']; del kwargs['cls']
        if not tabbed:
            params[name] = getattr(gr, cls_)(**kwargs)
        else:
            tab_name = kwargs['label'] if 'label' in kwargs else name
            with gr.Tab(tab_name):
                params[name] = getattr(gr, cls_)(**kwargs)
    return params

def _clear(session_state):
    session_state.clear()
    session_state.update(_default_session_state)
    return session_state

def _show_status(*args):
    kwargs = {k: v for k, v in zip(COMPONENTS.keys(), args)}
    return kwargs

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

################################################################
# Bot fn
################################################################

from utils.llms import _llm_call, _llm_call_stream, _random_bot_fn, _print_messages

def _openai_agent_bot_fn(message, history, **kwargs):
    # NOTE: use messages instead of history for advanced features (e.g. function calling), can not undo or retry
    messages = kwargs['session_state']['messages']
    chat_engine = 'gpt-4o'
    global TOOLS_SCHEMA
    tools_schema = [obj for obj in TOOLS_SCHEMA if obj['function']['name'] in kwargs['tools']]
    messages.append({'role': 'user', 'content': message})
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=chat_engine,
        messages=messages,
        tools = tools_schema,
        # **_kwargs,
    )
    bot_msg = resp.choices[0].message
    if bot_msg.tool_calls is None:
        bot_message = bot_msg.content
    else:
        tool_call = bot_msg.tool_calls[0]
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        try:
            result = getattr(TOOLS, function_name)(**arguments)
            
            function_call_desc = f"The function {function_name} was called with arguments {arguments}, returning result {result}."
            details = [{"title": f"🛠️ Use tool: {function_name}", "content": function_call_desc, "before": True}]

            if function_name in DIRECT_RESPONSE_TOOLS:
                bot_message = render_message(dict(text=TOOLS.format_direct_response(function_name, result, arguments), details=details))
            else:
                function_call_result_message = {
                    "role": "tool",
                    "content": json.dumps({
                        **arguments,
                        **TOOLS.format_returns(function_name, result)
                    }),
                    "tool_call_id": resp.choices[0].message.tool_calls[0].id
                }
                messages.append(bot_msg) # function_call_triggered_message
                messages.append(function_call_result_message)
                response = openai.chat.completions.create(
                    model=chat_engine,
                    messages=messages,
                )
                bot_message = render_message(dict(text=response.choices[0].message.content, details=details))
        except:
            bot_message = f'ERROR: Function call for {function_name} with arguments {arguments} failed.'
    messages.append({'role': 'assistant', 'content': bot_message})
    _print_messages(messages, tag=f':: openai_agent ({chat_engine})')
    return bot_message

def _langchain_agent_bot_fn(message, history, **kwargs):
    
    system_prompt = kwargs.get('system_prompt', None)
    session_state = kwargs['session_state']
    chat_engine = 'gpt-4o'

    messages = history
    if message:
        messages += [{'role': 'user', 'content': message}]
    if system_prompt:
        messages = [{'role': 'system', 'content': system_prompt}] + messages

    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(temperature=0, model=chat_engine)
    
    from utils.tools_langchain import get_tools
    tools = get_tools()

    from langchain.prompts import MessagesPlaceholder
    from langchain.memory import ConversationBufferMemory

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }

    if 'memory' not in session_state:
        session_state['memory'] = ConversationBufferMemory(memory_key="memory", return_messages=True)
    memory = session_state['memory']
    print(memory.load_memory_variables({}))

    mrkl = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True,
                agent_kwargs=agent_kwargs, memory=memory)
    bot_message = mrkl.invoke(message)['output']
    _print_messages(messages + [{'role': 'assistant', 'content': bot_message }], tag=f':: langchain_agent ({chat_engine})')
    return bot_message

def _slash_bot_fn(message, history, **kwargs):
    cmds = message[1:].split(' ', maxsplit=1)
    cmd, rest = cmds[0], cmds[1] if len(cmds) == 2 else ''
    return message

def bot_fn(message, history, *args):
    __TIC = time.time()
    kwargs = {k: v for k, v in zip(COMPONENTS.keys(), args)}

    session_state = kwargs['session_state']
    if len(history) == 0 or message == '/clear':
        _clear(session_state)
    session_state['previous_message'] = session_state['message']
    session_state['message'] = message
    # plain_message = _rerender_message(message)
    # history = _rerender_history(history[session_state['context_switch_at']:])

    ##########################################################

    # update "auto"
    AUTOS = {'chat_engine': 'gpt-3.5-turbo'}
    for param, default_value in AUTOS.items():
        kwargs[param] = default_value if kwargs[param] == 'auto' else kwargs[param]

    if message.startswith('/') or message.startswith('.'):
        bot_message = _slash_bot_fn(message, history, **kwargs)
    else:
        bot_message = {'random': _random_bot_fn,
            'gpt-3.5-turbo': _llm_call,
            'gpt-4o': _llm_call_stream,
            'openai_agent': _openai_agent_bot_fn,
            'langchain_agent': _langchain_agent_bot_fn,
            }.get(kwargs['chat_engine'])(message, history, **kwargs)
    
    ##########################################################
    
    if isinstance(bot_message, str):
        yield bot_message
    else:
        bot_message = yield from bot_message

    if kwargs.get('speech_synthesis', False):
        _speech_synthesis(_rerender_message(bot_message, format='speech'))
    __TOC = time.time()
    session_state['elapsed_time'] = __TOC - __TIC
    # print(json.dumps(kwargs, indent=2))
    print(pprint.pformat(kwargs))
    return bot_message

################################################################
# Demo
################################################################

def get_demo():
    global COMPONENTS, COMPONENTS_EXCLUDED
    css="""#chatbot {
    min-height: 600px;
    }
    .full-container label {
    display: block;
    padding-left: 8px;
    padding-right: 8px;
    }"""
    with gr.Blocks(css=css) as demo:
        # title
        gr.HTML(f"<center><h1>{TITLE}</h1></center>")
        # description
        with gr.Accordion("Expand to see Introduction and Usage", open=False):
            gr.Markdown(f"{DESCRIPTION}")
        with gr.Row():
            with gr.Column(scale=1):
                # settings
                for section_name, _settings in SETTINGS.items():
                    metadata = _settings['__metadata']
                    with gr.Accordion(section_name, open=metadata.get('open', False)):
                        settings = _create_from_dict(_settings, tabbed=metadata.get('tabbed', False))
                        COMPONENTS = {**COMPONENTS, **settings}
                COMPONENTS_EXCLUDED = {k: v for k, v in COMPONENTS.items() if k in EXCLUDED_KEYS}
                COMPONENTS = {k: v for k, v in COMPONENTS.items() if k not in EXCLUDED_KEYS}
            with gr.Column(scale=9):
                # chatbot
                from utils.utils import change_signature
                _sig_bot_fn = change_signature(['message', 'history'] + list(COMPONENTS.keys()))(bot_fn) # better API
                from utils.gradio import ChatInterface
                chatbot = ChatInterface(_sig_bot_fn, type='messages', 
                        additional_inputs=list(COMPONENTS.values()),
                        additional_outputs=[COMPONENTS['session_state'], COMPONENTS_EXCLUDED['status']],
                        multimodal=False,
                        avatar_images=('assets/user.png', 'assets/bot.png'))
                chatbot.audio_btn.click(transcribe, [], [chatbot.textbox], queue=False, api_name=False)
                COMPONENTS_EXCLUDED['show_status_btn'].click(_show_status, list(COMPONENTS.values()), [COMPONENTS_EXCLUDED['status']], api_name=False)
                # examples
                with gr.Accordion("Examples", open=False) as examples_accordin:
                    chat_examples = gr.Examples([
                        "What's the Everett interpretation of quantum mechanics?",
                        ], inputs=chatbot.textbox, label="AI Chat Examples",
                    )
    return demo

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

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    demo = get_demo()
    from utils.gradio import reload_javascript
    reload_javascript()
    demo.queue().launch(server_name='0.0.0.0', server_port=args.port)
