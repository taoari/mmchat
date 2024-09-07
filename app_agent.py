# app.py: Multimodal Chatbot
import os
import json
import time
import logging
import jinja2
import pprint
import gradio as gr

from utils.message import parse_message, render_message, _prefix_local_file
from dotenv import load_dotenv
from utils import prompts, llms
from utils.llms import _llm_call, _llm_call_stream, _random_bot_fn, _print_messages

# Load environment variables from .env file
load_dotenv()

# Default session state
_default_session_state = {
    'context_switch_at': 0,  # History before this point should be ignored (e.g., after uploading an image or file)
    'messages': [],
}

from utils import tools as TOOLS
import tool2schema
TOOLS_SCHEMA = tool2schema.FindToolEnabledSchemas(TOOLS)
print('Tools:\n' + json.dumps(TOOLS_SCHEMA, indent=2))

# DIRECT_RESPONSE_TOOLS = ['get_delivery_date'] # use function output directly instead of LLM rewrite
DIRECT_RESPONSE_TOOLS = [] # always rewrite function output with LLM
AVAILABLE_TOOLS = [obj["function"]["name"] for obj in TOOLS_SCHEMA if obj["type"] == "function"]

from app import SETTINGS
SETTINGS['Settings']['chat_engine'] = {
            'cls': 'Dropdown', 
            'choices': ['auto', 'random', 'gpt-3.5-turbo', 'gpt-4o', 'openai_agent', 'langchain_agent'], 
            'value': 'openai_agent', 
            'interactive': True, 
            'label': "Chat Engine"
        }

# Utility functions
def _clear(session_state):
    """Clear all session state keys except session_hash."""
    import copy
    session_hash = session_state.get('session_hash')
    session_state.clear()
    session_state.update(copy.deepcopy(_default_session_state))
    if session_hash:
        session_state['session_hash'] = session_hash
    return session_state

# Bot functions
def __helper_fn():
    import openai
    if True:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model_id = 'gpt-4o'
    else:
        client = openai.OpenAI(base_url='http://localhost:8080/v1', api_key='-')
        model_id = 'jsincn/phi-3-mini-128k-instruct-awq'
    return client, model_id

def _openai_agent_bot_fn(message, history, **kwargs):
    # NOTE: use messages instead of history for advanced features (e.g. function calling), can not undo or retry
    messages = kwargs['session_state']['messages']
    if len(messages) == 0:
        messages.append({'role': 'system', 'content': prompts.PROMPT_CHECK_DELIVERY_DATE})
    global TOOLS_SCHEMA
    tools_schema = [obj for obj in TOOLS_SCHEMA if obj['function']['name'] in kwargs['tools']]
    messages.append({'role': 'user', 'content': message})

    client, model_id = __helper_fn()

    try:
        # NOTE: TGI tools can have parse error in this call
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools = tools_schema,
            # **_kwargs,
        )
        bot_msg = resp.choices[0].message
    except:
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            # tools = tools_schema,
            # **_kwargs,
        )
        bot_msg = resp.choices[0].message

    # NOTE: openai is smart enough, if tools are not needed, then bot_msg.tool_calls is None
    # TGI tools support is primitive, always returns tool_calls, but with different function name and arguments.
    if bot_msg.tool_calls is None:
        bot_message = bot_msg.content
    else:
        tool_call = bot_msg.tool_calls[0]
        function_name = tool_call.function.name
        arguments = tool_call.function.arguments
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not hasattr(TOOLS, function_name): # also need to check signature match
            # NOTE: TGI tools only, means function call is not needed
            # need to apply to llm trained with LLM function call, otherwise, gives wrong name e.g CheckDeliveryName
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                # tools = tools_schema,
                # **_kwargs,
            )
            bot_message = resp.choices[0].message.content
        else:
            try:
                result = getattr(TOOLS, function_name)(**arguments)
                
                function_call_desc = f"The function {function_name} was called with arguments {arguments}, returning result {result}."
                details = [{"title": f"🧠 Use tool: {function_name}", "content": function_call_desc, "before": True}]

                if function_name in DIRECT_RESPONSE_TOOLS:
                    bot_message = render_message(dict(text=TOOLS.format_direct_response(function_name, result, arguments), details=details))
                else:
                    function_call_result_message = {"role": 'tool', "content": str(result), "tool_call_id": resp.choices[0].message.tool_calls[0].id}
                    messages.append(json.loads(bot_msg.to_json())) # function_call_triggered_message
                    messages.append(function_call_result_message)
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                    )
                    bot_message = render_message(dict(text=response.choices[0].message.content, details=details))
            except:
                bot_message = f'ERROR: Function call for {function_name} with arguments {arguments} failed.'
    messages.append({'role': 'assistant', 'content': bot_message})
    _print_messages(messages, tag=f':: openai_agent ({model_id})')
    return bot_message

def _langchain_agent_bot_fn(message, history, **kwargs):
    system_prompt = prompts.PROMPT_CHECK_DELIVERY_DATE
    session_state = kwargs['session_state']
    chat_engine = 'gpt-4o'

    messages = history
    if message:
        messages += [{'role': 'user', 'content': message}]
    if system_prompt:
        messages = [{'role': 'system', 'content': system_prompt}] + messages

    from langchain.agents import initialize_agent
    from langchain.agents import AgentType

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(temperature=0, model=chat_engine)
    
    from utils.tools_langchain import get_tools
    tools = get_tools()
    from langchain.memory import ConversationBufferMemory
    from langchain_core.prompts import MessagesPlaceholder

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
    """Handle bot commands starting with '/' or '.'."""
    cmd, *args = message[1:].split(' ', maxsplit=1)
    return message  # Command handling can be customized here

def bot_fn(message, history, **kwargs):
    """Main bot function to handle both commands and regular messages."""
    # Default "auto" behavior
    AUTOS = {'chat_engine': 'gpt-3.5-turbo'}
    for param, default_value in AUTOS.items():
        kwargs[param] = default_value if kwargs[param] == 'auto' else kwargs[param]

    # Clear session state if needed
    if not history or message == '/clear':
        _clear(kwargs['session_state'])

    # Handle commands or regular conversation
    if message.startswith(('/', '.')):
        bot_message = _slash_bot_fn(message, history, **kwargs)
    else:
        kwargs['tools'] = AVAILABLE_TOOLS
        bot_fn_map = {
            'random': _random_bot_fn,
            'openai_agent': _openai_agent_bot_fn,
            'langchain_agent': _langchain_agent_bot_fn,
        }
        bot_message = bot_fn_map.get(kwargs['chat_engine'], _llm_call_stream)(message, history, **kwargs)

    # Stream or yield bot message
    if isinstance(bot_message, str):
        yield bot_message
    else:
        yield from bot_message

# Argument parser
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

# Main entry point
if __name__ == '__main__':
    import app
    from app import main

    # Set bot function and parse arguments
    app.bot_fn = bot_fn
    args = parse_args()

    # Start the main app
    main(args)
