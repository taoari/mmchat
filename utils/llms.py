import os
import random
from utils.message import render_message, get_spinner

LLM_ENDPOINTS = {}

def parse_endpoints_from_environ():
    global LLM_ENDPOINTS
    for key, value in os.environ.items():
        if key.startswith('LLM_ENDPOINT_'):
            name = key[len('LLM_ENDPOINT_'):].lower()
            if ';' in value:
                url, model = value.split(';')
                LLM_ENDPOINTS[name.lower()] = dict(url=url, model=model)
            else:
                LLM_ENDPOINTS[name.lower()] = dict(url=value, model=name)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def _print_messages(messages, title='Chat history:', tag=""):
    from markdownify import markdownify
    from utils.utils import replace_extra_newlines
    icons = {'system': '🖥️', 'user': '👤', 'assistant': '🤖', 'tool': '🛠️', 'think': '🧠', 'message': '💬'}
    res = [] if title is None else [title]
    for message in messages:
        try:
            if 'tool_calls' in message and message['tool_calls']:
                res.append(f'{icons[message["role"]]}: {message["tool_calls"]}')
            else:
                res.append(f'{icons[message["role"]]}: {replace_extra_newlines(markdownify(message["content"]))}')
        except:
            res.append(f'{icons["message"]}: {message}') # think is assistant of function calling
    out_str = '\n'.join(res) + '\n'
    print(f"{bcolors.OKGREEN}{out_str}{bcolors.WARNING}{tag}{bcolors.ENDC}")

def _random_bot_fn(message, history, **kwargs):

    # Example multimodal messages
    samples = {}
    target = dict(text="I love cat", images=["https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"])
    samples['image'] = render_message(target)
    target = dict(audios=["https://upload.wikimedia.org/wikipedia/commons/2/28/Caldhu.wav"])
    samples['audio'] = render_message(target)
    target = dict(videos=["https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"])
    samples['video'] = render_message(target)
    target = dict(files=["https://www.africau.edu/images/default/sample.pdf"])
    samples['pdf'] = render_message(target)
    target = dict(text="Hello, how can I assist you today?", 
            buttons=['Primary', dict(text='Secondary', value="the second choice"), 
                    dict(text="More", href="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg")])
    samples['button'] = render_message(target)
    target = dict(text="We found the following items:", cards=[
        dict(image="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg", title="Siam Lilac Point", 
                text="The lilac point Siamese cat usually has a pale pink nose and pale pink paw pads.", buttons=[]),
        dict(image="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg", 
                title="Siam Lilac Point", text="The lilac point Siamese cat usually has a pale pink nose and pale pink paw pads.",
                buttons=[dict(text="Search", value="/search"),
                         dict(text="More", href="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg")])])
    samples['card'] = render_message(target)
    _message = get_spinner() + " Please be patient"
    samples['spinner'] = _message
    target = dict(text="This is a reference", references=[dict(title="Sources", sources=[
        dict(text="📁 hello.pdf", link="https://hello.com", score=0.5),
        dict(text="📁 World.pdf", link="https://world.com", score=0.3),
    ])])
    samples['reference'] = render_message(target)
    target = dict(text="Final results goes here", details=[dict(
            title="🛠️ Show progress", content="Scratch pad goes here", before=True)])
    samples['detail_before'] = render_message(target)
    target = dict(text="Final results goes here", details=[dict(
            title="🧠 Show progress", content="Scratch pad goes here", before=False)])
    samples['detail'] = render_message(target)
    samples['markdown'] = """
Hello **World**

![This is a cat](https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg)
    """
    samples['markdown_slack'] = """
Hello *World*

*Resources*
<https://hello.com|📁 hello.pdf> `score: 0.5`
    """

    if message in samples:
        bot_message = samples[message]
    elif message == 'all':
        bot_message = '<br >'.join(samples.values())
    else:
        bot_message = random.choice(list(samples.values()))
    messages = history + [{'role': 'user', 'content': message}]
    _print_messages(messages + [{'role': 'assistant', 'content': bot_message }], tag=":: random")
    return bot_message

def _get_llm(**kwargs):
    chat_engine = kwargs.get('chat_engine', 'gpt-4o-mini')
    import openai
    if chat_engine in LLM_ENDPOINTS:
        endpoint = LLM_ENDPOINTS[chat_engine]
        client = openai.OpenAI(base_url=f"{endpoint['url']}/v1", api_key='-')
        model_id = endpoint['model']
    else:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model_id = chat_engine
    return client, model_id

def _get_llm_langchain(**kwargs):
    _kwargs = dict(temperature=kwargs.get('temperature', 0))
    chat_engine = kwargs.get('chat_engine', 'gpt-4o-mini')

    from langchain_openai import ChatOpenAI
    if chat_engine in LLM_ENDPOINTS:
        endpoint = LLM_ENDPOINTS[chat_engine]
        llm = ChatOpenAI(
            model=endpoint['model'],
            openai_api_key="-",
            openai_api_base=f"{endpoint['url']}/v1",
            **_kwargs
        )
    else:
        llm = ChatOpenAI(model=chat_engine, **_kwargs)
    return llm
    
def _preprocess_messages(message, history, **kwargs):
    _kwargs = dict(temperature=kwargs.get('temperature', 0))
    system_prompt = kwargs.get('system_prompt', None)
    
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.extend(history)
    if message:
        messages.append({'role': 'user', 'content': message})
    return messages, _kwargs
    
def _llm_preprocess(message, history, **kwargs):
    client, model_id = _get_llm(**kwargs)
    messages, _kwargs = _preprocess_messages(message, history, **kwargs)
    return client, model_id, messages, _kwargs

def _llm_call(message, history, **kwargs):
    chat_engine = kwargs.get('chat_engine', 'gpt-4o-mini')
    client, model_id, messages, _kwargs = _llm_preprocess(message, history, **kwargs)
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        **_kwargs,
    )
    bot_message = resp.choices[0].message.content
    _print_messages(messages + [{'role': 'assistant', 'content': bot_message }], tag=f':: openai ({chat_engine})')
    if hasattr(resp, 'usage') and 'session_state' in kwargs:
        kwargs['session_state']['usage'] = resp.usage.__dict__
    return bot_message

def _llm_call_stream(message, history, **kwargs):
    chat_engine = kwargs.get('chat_engine', 'gpt-4o-mini')
    client, model_id, messages, _kwargs = _llm_preprocess(message, history, **kwargs)
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        stream=True,
        **_kwargs,
    )
    bot_message = ""
    for _resp in resp:
        if hasattr(_resp.choices[0].delta, 'content') and _resp.choices[0].delta.content:
            bot_message += _resp.choices[0].delta.content
            yield bot_message
    _print_messages(messages + [{'role': 'assistant', 'content': bot_message }], tag=f':: openai_stream ({chat_engine})')
    
    if 'session_state' in kwargs:
        from utils.utils import llm_count_tokens
        prompt_tokens = llm_count_tokens(messages)
        completion_tokens = llm_count_tokens([{'role': 'assistant', 'content': bot_message }])
        kwargs['session_state']['usage'] = {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 
                                            'total_tokens': prompt_tokens + completion_tokens}
    return bot_message