from functools import wraps
import inspect
import re
import tiktoken

def llm_count_tokens(messages, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    
    # Rough estimate: 3 tokens for every message's metadata (role, etc.)
    tokens_per_message = 3

    total_tokens = 0
    for message in messages:
        # Count tokens in the message content
        total_tokens += len(encoding.encode(message["content"]))
        # Add rough estimate for message metadata
        total_tokens += tokens_per_message

    return total_tokens

def replace_extra_newlines(text):
    # Replace more than two newlines (with possible whitespaces in between) with exactly two newlines
    return re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

def recursive_apply_to_dict_values(d, func):
    if isinstance(d, dict):
        return {k: recursive_apply_to_dict_values(v, func) for k, v in d.items()}
    elif isinstance(d, list):
        return [recursive_apply_to_dict_values(item, func) for item in d]
    else:
        return func(d)

def recursive_to_dict(obj, max_level=float('inf'), current_level=0):
    if current_level >= max_level:
        return str(obj)  # Limit reached, return a string representation or any other fallback

    if hasattr(obj, "__dict__"):
        return {key: recursive_to_dict(value, max_level, current_level + 1) for key, value in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [recursive_to_dict(item, max_level, current_level + 1) for item in obj]
    elif isinstance(obj, dict):
        return {key: recursive_to_dict(value, max_level, current_level + 1) for key, value in obj.items()}
    else:
        return obj

def change_signature(arg_list, kwarg_dict={}):
    def decorator(fn):
        # Create a signature from arg_list and kwarg_dict
        parameters = []
        for arg in arg_list:
            parameters.append(inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD))

        for kwarg, default in kwarg_dict.items():
            parameters.append(inspect.Parameter(kwarg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default))

        new_signature = inspect.Signature(parameters)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            bound_args = new_signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            result = fn(*bound_args.args, **bound_args.kwargs)
            
            # Handle generator functions
            if inspect.isgeneratorfunction(fn):
                yield from result
            else:
                return result

        wrapper.__signature__ = new_signature
        return wrapper
    return decorator

def get_temp_file_name(prefix='gradio/app-', suffix='', filename=None):
    import os
    import tempfile
    if filename is not None:
        fname = os.path.join(tempfile.gettempdir(), filename)
    else:
        fname = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix).name
    return fname