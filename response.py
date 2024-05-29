import requests
import json
from discord import Message
import nba_scraper
import chat_request
import weather


def get_quote() -> str:
    response = requests.get('https://zenquotes.io/api/random')
    json_data = json.loads(response.text)
    quote = '"' + json_data[0]['q'] + '"\n -' + json_data[0]['a']
    return quote    

def get_response(message: Message, user_input: str) -> str:
    if (user_input == 'quote'):
        return get_quote()
    elif (user_input == 'nba'):
        return nba_scraper.get_nba_score()
    elif (user_input.startswith('chat')):
        return chat_request.get_chat_response(user_input[5:])
    elif (user_input == 'weather'):
        return weather.get_weather(user_input[8:])
    elif (user_input == 'help'):
        return 'Available commands: quote, nba, chat, weather, help'
    else:
        return 'bozo'