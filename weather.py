import aiohttp
import os
from dotenv import load_dotenv
import nextcord

load_dotenv()

KEY = os.getenv('WEATHER-API')

async def get_weather(message, location: str) -> str:
    if location:
        url = 'http://api.weatherapi.com/v1/current.json'
        params = {
            'key': KEY,
            'q': location
        }
        
        async with aiohttp.ClientSession as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                location = data['location']['name']
                temp_c = data['current']['temp_c']
                temp_f = data['current']['temp_f']
                humidity = data['current']['humidity']
                wind_kph = data['current']['wind_kph']
                wind_mph = data['current']['wind_mph']
                condition = data['current']['condition']['text']
                precipitation = data['current']['precip_mm']
                image_url ='http:' + data['current']['condition']['icon']
                
                embed = nextcord.Embed(title=f'Weather in {location}', description=f'The condition in `{location}` is `{condition}`', color=0x00ff00)
                embed.add_field(name='Temperature', value=f'{temp_c}°C | {temp_f}°F')
                embed.add_field(name='Humidity', value=f'{humidity}%')
                embed.add_field(name='Wind Speeds', value=f'{wind_kph} kph | {wind_mph} mph')
                embed.add_field(name='Precipitation', value=f'{precipitation} mm')
                embed.set_thumbnail(url=image_url) 
                
                return embed
    else:
        return 'Please provide a location'