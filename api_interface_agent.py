from gradio_client import Client

client = Client("http://localhost:7860/")

def gradio_chat(message):
    result = client.predict(
            message=message,
            system_prompt="You are a helpful customer support assistant. Use the supplied tools to assist the user.",
            chat_engine="agent",
            speech_synthesis=False,
            temperature=0.7,
            api_name="/chat"
    )
    return result

def interactive_chatbot():
    print('This is an interactive chatbot powered by Gradio. Type "/exit" to stop.')
    while True:
        message = input('👤: ')
        if message == '/exit':
            break
        print(f'🤖: {gradio_chat(message)}')

def test_conversation(user_inputs):
    for message in user_inputs:
        # system: ⚙
        print(f'👤: {message}')
        print(f'🤖: {gradio_chat(message)}')

if __name__ == '__main__':

    test_conversation(['Hello',
        'Can you tell me the delivery date for my order?', 
        '12345'])

    # interactive_chatbot()
    # python -c "import api_interface; api_interface.interactive_chatbot()"