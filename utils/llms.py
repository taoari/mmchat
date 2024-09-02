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

parse_endpoints_from_environ()

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
    icons = {'system': '🖥️', 'user': '👤', 'assistant': '🤖'}
    res = [] if title is None else [title]
    for message in messages:
        res.append(f'{icons[message["role"]]}: {message["content"]}')
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

def __llm_call_preprocess(message, history, **kwargs):
    _kwargs = dict(temperature=max(0.001, kwargs.get('temperature', 0.001)), 
                   max_tokens=kwargs.get('max_tokens', 1024))
    system_prompt = kwargs.get('system_prompt', None)
    chat_engine = kwargs.get('chat_engine', 'gpt-3.5-turbo')

    messages = history
    if message:
        messages += [{'role': 'user', 'content': message}]
    if system_prompt:
        messages = [{'role': 'system', 'content': system_prompt}] + messages

    import openai
    if chat_engine in LLM_ENDPOINTS:
        endpoint = LLM_ENDPOINTS[chat_engine]
        client = openai.OpenAI(base_url=f"{endpoint['url']}/v1", api_key='-')
        model_id = endpoint['model']
    else:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model_id = chat_engine

    return client, model_id, messages, _kwargs

def _llm_call(message, history, **kwargs):
    chat_engine = kwargs.get('chat_engine', 'gpt-3.5-turbo')
    client, model_id, messages, _kwargs = __llm_call_preprocess(message, history, **kwargs)
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        **_kwargs,
    )
    bot_message = resp.choices[0].message.content
    _print_messages(messages + [{'role': 'assistant', 'content': bot_message }], tag=f':: openai ({chat_engine})')
    return bot_message

def _llm_call_stream(message, history, **kwargs):
    chat_engine = kwargs.get('chat_engine', 'gpt-3.5-turbo')
    client, model_id, messages, _kwargs = __llm_call_preprocess(message, history, **kwargs)
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
    return bot_message