from discord import Intents, Client, Message
import requests
from nba_scraper import get_nba_score
import json
from news import get_cnn

def get_quote() -> str:
    response = requests.get('https://zenquotes.io/api/random')
    json_data = json.loads(response.text)
    quote = '"' + json_data[0]['q'] + '"\n -' + json_data[0]['a']
    return quote    

def get_response(message: Message, user_input: str) -> str:
    if (user_input == 'quote'):
        return get_quote()
    elif (user_input == 'nba'):
        return get_nba_score()
    # elif (user_input == 'news'):
    #     return get_title()
    elif (user_input == 'help'):
        return 'Available commands: quote, nba, news, help'
    else:
        return 'bozo'
    
def get_diff_news(news_source: str) -> str:
    if (news_source == "cnn"):
        return get_cnn()
        #TODO: add more news sources (like 2 more)
    else:
        return "Pablo can't handle any more news.."

