import os, uuid
from gradio_client import Client

# client = Client(os.environ.get('GRADIO_SERVER'))
client = Client("http://localhost:7860/demo")

def gradio_chat(message, session_hash=None):
    client.session_hash = session_hash if session_hash is not None else str(uuid.uuid4())
    
    result = client.predict(
            message,
            api_name="/chat"
    )
    return result

def interactive_chatbot():
    print('This is an interactive chatbot powered by Gradio. Type "/exit" to stop.')
    while True:
        message = input('👤: ')
        if message == '/exit':
            break
        print(f'🤖: {gradio_chat(message, session_hash="default")}')

def test_conversation(user_inputs):
    for message in user_inputs:
        # system: ⚙
        print(f'👤: {message}')
        print(f'🤖: {gradio_chat(message, session_hash="default")}')

if __name__ == '__main__':

    test_conversation(['Hello',
        'My name is Alice', 
        'What is my name?'])

    # interactive_chatbot()
    # python -c "import api_interface; api_interface.interactive_chatbot()"