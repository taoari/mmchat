# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount

import os, uuid
from dotenv import load_dotenv
from gradio_client import Client

load_dotenv()

client = Client(os.environ.get('GRADIO_SERVER'))

def gradio_chat(message, session_hash=None):
    client.session_hash = session_hash if session_hash is not None else str(uuid.uuid4())
    
    result = client.predict(
            message,
            api_name="/chat"
    )
    return result

class MyBot(ActivityHandler):
    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.

    async def on_message_activity(self, turn_context: TurnContext):
        # await turn_context.send_activity(f"You said '{ turn_context.activity.text }'")
        message = turn_context.activity.text
        conversation_id = turn_context.activity.conversation.id
        await turn_context.send_activity(f"{gradio_chat(message, conversation_id)}")
        

    async def on_members_added_activity(
        self,
        members_added: ChannelAccount,
        turn_context: TurnContext
    ):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello and welcome!")
