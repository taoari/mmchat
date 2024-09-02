import inspect
from functools import wraps
import re
import gradio as gr

from utils.gradio import ChatInterface, reload_javascript

def _convert_history(history, type='tuples'):
    assert type in ['tuples', 'messages']
    if len(history) > 0:
        history_type = 'messages' if isinstance(history[0], dict) else 'tuples'
        if history_type == type:
            return history
        
        if history_type == 'messages':
            res = []
            for msg in history:
                if msg['role'] == 'user':
                    res.append((msg, None))
                else:
                    res.append((None, msg))
        else:
            for user, bot in history:
                if user is not None:
                    res.append({'role': 'user', 'content': user})
                if bot is not None:
                    res.append({'role': 'assistant', 'content': bot})
    return res
    
class AgentChatInterface(ChatInterface):

    def _setup_event(self):
        textbox, chatbot, chatbot_state, additional_inputs = self.textbox, self.chatbot, self.chatbot_state, self.additional_inputs
        submit_btn, retry_btn, undo_btn, clear_btn, upload_btn = self.submit_btn, self.retry_btn, self.undo_btn, self.clear_btn, self.upload_btn
        fake_response, fake_api_btn = self.fake_response, self.fake_api_btn
        export_btn, export_btn_hidden = self.export_btn, self.export_btn_hidden

        # self._setup_api_fn(fake_api_btn.click, textbox, chatbot_state, fake_response, additional_inputs)
        self._setup_submit(textbox.submit, textbox, chatbot, additional_inputs, api_name='chat_with_history')
        self._setup_submit(submit_btn.click, textbox, chatbot, additional_inputs, api_name=False)

        retry_btn.click(self._retry, [chatbot] + additional_inputs, [chatbot], api_name=False)
        undo_btn.click(self._undo, [textbox, chatbot], [textbox, chatbot], api_name=False) # NOTE: state can not undo or retry
        clear_btn.click(self._clear, None, [chatbot, chatbot_state], api_name=False)

        upload_btn.upload(
            self._upload_fn, 
            [textbox, upload_btn], 
            [textbox], queue=False, api_name='upload')
        export_btn.click(self._export, [chatbot], [export_btn_hidden], api_name=False).then(
            fn=None, inputs=None, outputs=None, js="() => document.querySelector('#export_btn_hidden').click()")

    def _setup_submit(self, event_trigger, textbox, chatbot, additional_inputs, api_name='chat_with_history'):
        bot_msg = event_trigger(self.fn, [textbox, chatbot] + additional_inputs, [chatbot], api_name=api_name)
        bot_msg.then(lambda: gr.update(interactive=True, value=None), None, [textbox], api_name=False)
    
    def _retry(self, history, *args):
        if len(history) >= 2:
            message = history[-2]['content']
            _history = history[:-2]
            if self.is_generator:
                # clear history first
                yield history[:-1]

                yield from self.fn(message, _history, *args)
            else:
                yield self.fn(message, _history, *args)
        else:
            yield history

if __name__ == "__main__":

    # python -m utils.agent
    from utils.llms import _llm_call, _llm_call_stream
    def agent_bot_fn(message, history, *args):
        history += [{'role': 'user', 'content': message}]
        yield history
        bot_message = _llm_call_stream(None, history)
        for _bot_message in bot_message:
            _history = history + [{'role': 'assistant', 'content': _bot_message}]
            yield _history
    
    with gr.Blocks() as demo:
        chatbot = AgentChatInterface(agent_bot_fn, type='messages')
    reload_javascript()
    demo.queue().launch(server_name='0.0.0.0')