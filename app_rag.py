# app.py: Multimodal Chatbot
import os
import json
import time
import logging
import jinja2
import pprint
import gradio as gr

from utils.message import parse_message, render_message, _prefix_local_file, _rerender_message, _rerender_history
from dotenv import load_dotenv
from utils import prompts, llms
from utils.llms import _random_bot_fn
from langchain.globals import set_debug

set_debug(True)

REPHRASE_THRESH = 0.2

# Load environment variables from .env file
load_dotenv()

if os.environ.get('LLM_TRACING') == 'phoenix':
    import phoenix as px
    from phoenix.trace.langchain import LangChainInstrumentor
    os.environ['PHOENIX_PROJECT_NAME'] = "app-rag"
    # Initialize Langchain auto-instrumentation
    LangChainInstrumentor().instrument()

# Default session state
_default_session_state = {
    'context_switch_at': 0,  # History before this point should be ignored (e.g., after uploading an image or file)
    'messages': [],
    'bot_fn': None, # overwrite bot_fn
}
CACHE = {"vectorstores": {}, "clients": {}}

from app import SETTINGS
SETTINGS['Parameters']['bot_fn'] = {
            'cls': 'Dropdown', 
            'choices': ['auto', 'llm', 'search', 'rag', 'rag_intent', 'rag_rewrite_retrieve_read', 'rag_rewrite_retrieve_read_search'], 
            'value': 'rag_intent',
            'label': "Bot Function",
        }
SETTINGS['Parameters']['query_k'] = {'cls': 'Slider', 'minimum': 1, 'maximum': 5, 'value': 3, 'step': 1, 'label': "Number of sources"}

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

def setup_vectorstores(args):
    """setup vectorstores collection based on provided arguments."""
    from utils.vectorstore2 import get_vectordb, build_vectordb

    for collection in args.collection_name:
        if collection == 'intents':
            # Use Chroma (in-memory for the "intents" collection)
            vectordb = build_vectordb('chroma', collection, 'data/intents/nlu.json', embeddings_conn_str=args.embeddings)
        elif args.vectorstore == "chroma":
            vectordb = build_vectordb(args.vectorstore, collection, f'data/collections/{collection}', embeddings_conn_str=args.embeddings)
        else:
            vectordb = get_vectordb(args.vectorstore, collection, embeddings_conn_str=args.embeddings)
        CACHE['vectorstores'][collection] = vectordb

def format_document(doc, score):
    """Format document for display."""
    metadata = doc.metadata
    title = metadata.get('name') or metadata.get('title', None) # null-coalescing
    file_ref = metadata['source'] if 'page' not in metadata else f"{metadata['source']}#page={metadata['page'] + 1}"
    file_server = os.environ.get('FILE_SERVER', '')
    if file_server:
        link = f"{file_server.rstrip('/')}/{file_ref.lstrip('/')}"
    elif args.env == 'prod_fastapi':
        link = _prefix_local_file(file_ref, args.mount_path)
    else:
        link = _prefix_local_file(file_ref)
    return {
        'text': f"📁 {title if title else os.path.basename(file_ref)}", 
        'link': link, 
        'score': score
    }

# Bot functions
def _search_bot_fn(message, history, **kwargs):
    # NOTE: only implemented for elasticsearch
    vectorstore = args.vectorstore
    if vectorstore != 'elasticsearch':
        return "Search functionality is only implemented for elasticsearch."
    
    from utils.search import get_client, search

    if vectorstore not in CACHE['clients']:
        
        client = get_client(vectorstore)
        CACHE[vectorstore] = client
    else:
        client = CACHE[vectorstore]

    results = search(client, args.collection_name, message)['hits']['hits']

    from langchain_core.documents import Document
    docs = [Document(page_content=res['_source']['text'], metadata=res['_source']['metadata']) for res in results]
    scores = [res['_score'] for res in results]

    sources = [format_document(doc, score) for doc, score in zip(docs, scores)]
    return render_message({'references': [{'title': "Sources", 'sources': sources}]})

def _rag_bot_fn(message, history, **kwargs):
    """RAG-based bot response function."""
    collection = kwargs.get('collection', args.collection_name[0])
    chat_engine = kwargs['chat_engine']
    vectordb = CACHE['vectorstores'][collection]

    # Perform similarity search
    docs_with_scores = vectordb.similarity_search_with_score(message, k=kwargs.get('query_k', 3))
    docs = [doc for doc, score in docs_with_scores]
    if args.vectorstore in ['chroma', 'pgvector']:
        scores = [1.0 - score for _, score in docs_with_scores]
    else:
        scores = [score for _, score in docs_with_scores]
    sources = [format_document(doc, score) for doc, score in zip(docs, scores)]

    # LLM response with RAG system prompt
    system_prompt = jinja2.Template(prompts.PROMPT_RAG).render(docs=docs)
    _kwargs = {**kwargs, 'system_prompt': system_prompt}
    # NOTE: use messages and json for API, use history (persistence) and html for UI
    _history = _rerender_history(history, 'plain')
    bot_response = llms._llm_call_stream(message, _history, **_kwargs)

    for chunk in bot_response:
        yield render_message({'text': chunk, 'references': [{'title': "Sources", 'sources': sources}]})

def _similarity_search_with_score(vectordb, message, **kwargs):
    docs_with_scores = vectordb.similarity_search_with_score(message, k=kwargs.get('query_k', 3))
    if vectordb.__class__.__name__ in ['Chroma']:
        # NOTE: chroma, pgvector returns distance, elasticsearch return similarity
        docs_with_scores = [(doc, 1.0 - score) for doc, score in docs_with_scores]
    return docs_with_scores

def similarity_search_with_scores(message, **kwargs):
    docs_with_scores = []
    for collection in args.collection_name:
        vectordb = CACHE['vectorstores'][collection]
        docs_with_scores.extend(_similarity_search_with_score(vectordb, message, **kwargs))
    return sorted(docs_with_scores, key=lambda x: x[1], reverse=True)

def _intent_classif(docs_with_scores):
    """Return intent if best match is from 'intents' collection else None."""
    intent = None
    for doc, score in docs_with_scores:
        if score < REPHRASE_THRESH:
            intent = 'rephrase'
        elif 'intent' in doc.metadata:
            intent = doc.metadata['intent']
        break
    return intent

def _rag_bot_fn_with_intent_classification(message, history, **kwargs):
    """RAG-based bot response function."""

    # Perform similarity search
    k = kwargs.get('query_k', 3)
    docs_with_scores = similarity_search_with_scores(message, **kwargs)
    
    intent = _intent_classif(docs_with_scores)
    kwargs['session_state']['intent'] = intent

    if intent == 'rephrase':
        yield "Please rephrase your question."
    elif intent is not None:
        # TODO: use if intent == "<intent>" for different intents
        yield from llms._llm_call_stream(message, history, **kwargs)
    else:
        docs_with_scores = [(doc, score) for doc, score in docs_with_scores if 'intent' not in doc.metadata]
        docs_with_scores = docs_with_scores[:k]

        docs = [doc for doc, score in docs_with_scores]
        scores = [score for _, score in docs_with_scores]
        sources = [format_document(doc, score) for doc, score in zip(docs, scores)]

        # LLM response with RAG system prompt
        system_prompt = jinja2.Template(prompts.PROMPT_RAG).render(docs=docs)
        _kwargs = {**kwargs, 'system_prompt': system_prompt}
        # NOTE: use messages and json for API, use history (persistence) and html for UI
        _history = _rerender_history(history, 'plain')
        bot_response = llms._llm_call_stream(message, _history, **_kwargs)

        for chunk in bot_response:
            yield render_message({'text': chunk, 'references': [{'title': "Sources", 'sources': sources}]})

def _rag_rewrite_retrieve_read_search(message, history, **kwargs):
    from utils.bot_fn import rewrite_retrieval_read
    res = {}
    model = llms._get_llm_langchain(**kwargs)
    rewrite_retrieval_read(message, model=model, res=res)
    return render_message({
        'text': res['output'],
        'details': [{'title': "🛠️ Query rewrite", 'content': res['rewrite'], 'before': True}],
    })

def _rag_rewrite_retrieve_read(message, history, **kwargs):
    collection = kwargs.get('collection', 'default')
    chat_engine = kwargs['chat_engine']
    vectordb = CACHE['vectorstores'][collection]
    res = {}

    def retriever(query):
        docs_with_scores = vectordb.similarity_search_with_score(query, k=kwargs.get('query_k', 3))
        docs = [doc for doc, score in docs_with_scores]
        scores = [1.0 - score for _, score in docs_with_scores]
        res.update({"docs": docs, "scores": scores})
        return '\n\n'.join([doc.page_content for doc in docs])
    
    model = llms._get_llm_langchain(**kwargs)
    from utils.bot_fn import rewrite_retrieval_read
    rewrite_retrieval_read(message, retriever=retriever, model=model, res=res)

    sources = [format_document(doc, score) for doc, score in zip(res['docs'], res['scores'])]
    return render_message({
        'text': res['output'],
        'details': [{'title': "🛠️ Query rewrite", 'content': res['rewrite'], 'before': True}],
        'references': [{'title': "Sources", 'sources': sources}],
    })

def _slash_bot_fn(message, history, **kwargs):
    """Handle bot commands starting with '/' or '.'."""
    cmd, *args = message[1:].split(' ', maxsplit=1)
    if cmd in ['search', 'rag']:
        kwargs['session_state']['bot_fn'] = cmd
        bot_message = f'Set bot_fn to **{cmd}**.'
    else:
        bot_message = f'You said: {message}'
    return bot_message  # Command handling can be customized here

def bot_fn(message, history, **kwargs):
    """Main bot function to handle both commands and regular messages."""
    # Default "auto" behavior
    AUTOS = {'bot_fn': 'llm', 'chat_engine': 'gpt-4o-mini', 'format': 'html'}
    for param, default_value in AUTOS.items():
        kwargs[param] = default_value if kwargs[param] == 'auto' else kwargs[param]

    # Clear session state if needed
    if not history or message == '/clear':
        _clear(kwargs['session_state'])

    # NOTE: maintain and use messages instead of history if bot reponse is not simple text
    messages = kwargs['session_state']['messages']
    
    # Handle commands or regular conversation
    if message.startswith(('/', '.')):
        bot_message = _slash_bot_fn(message, messages, **kwargs)
    else:
        bot_fn_map = {
            'random': _random_bot_fn,
            'search': _search_bot_fn,
            'rag': _rag_bot_fn,
            'rag_intent': _rag_bot_fn_with_intent_classification,
            'rag_rewrite_retrieve_read': _rag_rewrite_retrieve_read,
            'rag_rewrite_retrieve_read_search': _rag_rewrite_retrieve_read_search,
        }
        kwargs['bot_fn'] = kwargs['session_state']['bot_fn'] if kwargs['session_state']['bot_fn'] is not None else kwargs['bot_fn']
        bot_message = bot_fn_map.get(kwargs['bot_fn'], llms._llm_call_stream)(message, history, **kwargs)

    # Stream or yield bot message
    if isinstance(bot_message, str):
        yield _rerender_message(bot_message, kwargs['format'])
    else:
        for _bot_msg in bot_message:
            yield _rerender_message(_bot_msg, kwargs['format'])
        bot_message = _bot_msg

    # messages.append({'role': 'user', 'content': message})
    # messages.append({'role': 'assistant', 'content': bot_message})
    return bot_message

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
    parser.add_argument(
        '--open-in-webbrowser', action='store_true',
        help='Open in web browser.')
    parser.add_argument('--autogen-yaml', action='store_true', 
        help='Auto-generate YAML files for PDF documents.')
    parser.add_argument('-vs', '--vectorstore', default='elasticsearch', 
        help='Vectorstore type')
    parser.add_argument('-c', '--collection-name', default='workplace', 
        help='collection name')
    parser.add_argument('--embeddings', default=None, type=str, 
        help='embeddings (format: "<endpoint>;<model_id>")')

    args = parser.parse_args()
    return args

# Main entry point
if __name__ == '__main__':
    import app
    from app import main

    # Set bot function and parse arguments
    app._default_session_state = _default_session_state
    app.bot_fn = bot_fn
    args = parse_args()
    if 'COLLECTION_NAME' in os.environ:
        args.collection_name = os.environ['COLLECTION_NAME'].split(',')
    if 'EMBEDDINGS_CONNECTION_STRING' in os.environ:
        args.embeddings = os.environ['EMBEDDINGS_CONNECTION_STRING']
    print(args)

    # setup vectorstores and configure Gradio static paths
    setup_vectorstores(args)
    gr.set_static_paths(paths=["data/collections"])

    # Start the main app
    main(args)
