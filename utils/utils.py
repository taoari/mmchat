from functools import wraps
import inspect
import re

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
    
def recursive_to_dict(v):
    def _convert(item):
        if hasattr(item, '__dict__'):
            return item.__dict__
        return item
    return recursive_apply_to_dict_values(v, _convert)

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