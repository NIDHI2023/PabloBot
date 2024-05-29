import discord
import logging
import os

from dotenv import load_dotenv
from response import get_response, get_diff_news
from discord import Client, Intents, Message


load_dotenv()
TOKEN = os.getenv('TOKEN')


intents = Intents.default()
intents.message_content = True
client = Client(intents=intents)

async def send_response(message: Message, user_message: str) -> None:
    class NewsButtons(discord.ui.View):
        def __init__(self):
            super().__init__()
        
        @discord.ui.button(label="CNN", style=discord.ButtonStyle.blurple)
        async def cnnBtn(self, interaction: discord.Interaction, button: discord.ui.Button):
            await interaction.response.send_message(get_diff_news("cnn"))
        
        @discord.ui.button(label="NBC", style=discord.ButtonStyle.blurple)
        async def nbcBtn(self, interaction: discord.Interaction, button: discord.ui.Button):
            await interaction.response.send_message(get_diff_news("nbc"))

        @discord.ui.button(style=discord.ButtonStyle.blurple, 
        emoji="🔥")
        async def emojiBtn(self, interaction: discord.Interaction, button: discord.ui.Button):
            await interaction.response.send_message(get_diff_news(""))


    try:
        if user_message[0] == '?':
            user_message = user_message[1:]
            response = await get_response(message, user_message)
            if user_message == "news":
                await message.author.send(view = NewsButtons())
            elif user_message == "weather":
                await message.author.send(embed=response)
            else:
                await message.author.send(response)
        elif user_message[0] == '!':
            user_message = user_message[1:]
            response = await get_response(message, user_message)
            
            if user_message.startswith("nba"):
                if type(response) == discord.Embed:
                    await message.channel.send(embed=response)
                else:
                    await message.channel.send(response)
            elif user_message == "news":
                await message.channel.send(view = NewsButtons())
            elif user_message.startswith("weather"):
                await message.channel.send(embed=response)
            else:
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