from discord import Intents, Client, Message
import os
from dotenv import load_dotenv
from response import get_response
import logging

load_dotenv()
TOKEN = os.getenv('TOKEN')


intents = Intents.default()
intents.message_content = True
client = Client(intents=intents)

async def send_response(message: Message, user_message: str) -> None:
    try:
        if user_message[0] == '?':
            user_message = user_message[1:]
            response = get_response(message, user_message)
            await message.author.send(response)
        elif user_message[0] == '!':
            user_message = user_message[1:]
            response = get_response(message, user_message)
            await message.channel.send(response)
        else:
            return
    except Exception as e:
        await message.channel.send(f"An error occured: {e}")
    
@client.event
async def on_ready() -> None:
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_message(message: Message) -> None:
    if message.author == client.user:
        return
    
    content = str(message.content)
    username = str(message.author)
    channel = str(message.channel)

    logger = logging.getLogger('discord')
    logger.setLevel(logging.INFO)
    logging.info(f'Message from {username} in {channel}: "{content}"')
    
    await send_response(message, message.content)
    
def main():
    client.run(token=TOKEN)  
    
if __name__ == '__main__':
    main()


