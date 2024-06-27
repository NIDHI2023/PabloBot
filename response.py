import requests
import json
import news
import nba_scraper
import chat_request
import weather
import twitch

from discord import Message

def get_quote() -> str:
    response = requests.get('https://zenquotes.io/api/random')
    json_data = json.loads(response.text)
    quote = '"' + json_data[0]['q'] + '"\n -' + json_data[0]['a']
    return quote    

async def get_response(message: Message, user_input: str):
    if user_input == 'quote':
        return get_quote()
    elif user_input.startswith('nba'):
        if user_input[4:] != '':
            return nba_scraper.get_nba_score(user_input[4:])
        else:
            return nba_scraper.get_nba_score()
    elif user_input.startswith('chat'):
        return chat_request.generate(user_input[5:])
    elif user_input.startswith('weather'):
        return await weather.get_weather(message, user_input[8:])
    elif user_input.startswith('twitch'):
        if len(user_input.split(" ")) != 2:
            return "Use: !twitch <channel name>. Channel must be live (or in database ig)."
        return await twitch.get_message(user_input.split(" ")[1])
    elif user_input == 'help':
        return 'Available commands: quote, nba, news, chat, weather, twitch, help'
    else:
        return 'bozo'
                                     
def get_diff_news(news_source: str) -> str:
    if news_source == "cnn":
        return news.get_cnn()
    elif news_source == "nbc":
        return news.get_nbc()
    else:
        return "Pablo can't handle any more news.."

