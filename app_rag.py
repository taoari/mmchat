# app.py : Multimodal Chatbot
import os
import json
import time
import logging
import jinja2
import pprint
import gradio as gr

from utils.message import parse_message, render_message, _prefix_local_file

################################################################
# Load .env and logging
################################################################

from dotenv import load_dotenv
from utils import prompts
from utils import llms
from utils.llms import _llm_call, _llm_call_stream, _random_bot_fn, _print_messages

load_dotenv()  # take environment variables from .env.

################################################################
# Globals
################################################################
_default_session_state = {
    'context_switch_at': 0,  # History before this point should be ignored (e.g., after uploading an image or file)
    'message': None,
    'previous_message': None,
}

CACHE = {"vectorstores": {}}

################################################################
# Utils
################################################################

def _clear(session_state):
    """Clear all session state keys except session_hash."""
    import copy
    session_hash = session_state.get('session_hash')
    session_state.clear()
    session_state.update(copy.deepcopy(_default_session_state))
    if session_hash:
        session_state['session_hash'] = session_hash
    return session_state

def prebuild_vectorstores(args):
    from utils.vectorstore import _build_vs_collection
    autogen_yaml = args.autogen_yaml
    CACHE['vectorstores']['default'] = _build_vs_collection('data/collections/default', 'default', 
            autogen_yaml=autogen_yaml, verbose=True)
    
def _format_doc(doc, score):
    _f = f"{doc.metadata['source']}#page={doc.metadata['page'] + 1}"
    path = '/demo' if args.env == 'prod_fastapi' else None
    return dict(text=f"📁 {os.path.split(_f)[-1]}", link=_prefix_local_file(_f, path), score=score)

################################################################
# Bot fn
################################################################

def _rag_bot_fn(message, history, **kwargs):
    collection = kwargs.get('collection', 'default')
    chat_engine = 'gpt-3.5-turbo'
    vectordb = CACHE['vectorstores'][collection]

    # similarity search
    docs = vectordb.similarity_search_with_score(message, k=3)
    scores = [1.0 - _r[1] for _r in docs] # extract scores
    docs = [_r[0] for _r in docs]

    # llm rag
    system_prompt = jinja2.Template(prompts.PROMPT_RAG).render(docs=docs)
    _kwargs = {**kwargs, 'chat_engine': chat_engine, 'system_prompt': system_prompt} # overwrite system_prompt
    sources = [_format_doc(doc, score) for doc, score in zip(docs, scores)]
    bot_message = llms._llm_call_stream(message, history, **_kwargs)
    for chunk in bot_message:
        yield render_message({'text': chunk, 'references': [dict(title="Sources", sources=sources)]})

def _slash_bot_fn(message, history, **kwargs):
    cmd, *rest = message[1:].split(' ', maxsplit=1)
    return message

def bot_fn(message, history, **kwargs):
    # Default "auto" behavior
    AUTOS = {'chat_engine': 'gpt-3.5-turbo'}
    for param, default_value in AUTOS.items():
        kwargs[param] = default_value if kwargs[param] == 'auto' else kwargs[param]

    if not history or message == '/clear':
        _clear(kwargs['session_state'])

    if message.startswith('/') or message.startswith('.'):
        bot_message = _slash_bot_fn(message, history, **kwargs)
    else:
        bot_message = _rag_bot_fn(message, history, **kwargs)

    if isinstance(bot_message, str):
        yield bot_message
    else:
        yield from bot_message
    return bot_message

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
        '--autogen-yaml', action='store_true', 
        help='auto generate yaml files for PDF files.')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import app
    from app import main
    # app.SETTINGS = SETTINGS
    app.bot_fn = bot_fn

    args = parse_args()

    prebuild_vectorstores(args)
    gr.set_static_paths(paths=["data/collections"])

    main(args)
