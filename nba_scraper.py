import time
import discord

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from bs4 import BeautifulSoup


# Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless") 
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--log-level=3")  

caps = DesiredCapabilities.CHROME
caps['goog:loggingPrefs'] = {'browser': 'OFF', 'performance': 'OFF'}
driver = webdriver.Chrome(options=chrome_options)

curr_date = time.strftime('%Y%m%d')

def get_nba_score(date = curr_date) -> str:
    if date != curr_date:
        t = date.split('/')
        if len(t) != 3 or int(t[0]) > 12 or int(t[0]) < 1 or int(t[1]) > 31 or int(t[1]) < 1 or int(t[2]) < 2020:
            return 'Invalid date or format. Please use the format MM/DD/YYYY and a date after Jan 1, 2020.'
        
        for i in range(len(t)):
            if len(t[i]) == 1:
                t[i] = '0' + t[i]
        date = t[2] + t[0] + t[1]
    url = f'https://www.espn.com/nba/scoreboard/_/date/{date}'
    # Scrape the page
    driver.get(url)
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # FIX THIS LATER ------------------------------------------------------------------------------------------------------------
    teams = soup.find_all('div', class_='ScoreCell__TeamName ScoreCell__TeamName--shortDisplayName truncate db')
    scores = soup.find_all('div', class_='ScoreCell__Score h4 clr-gray-01 fw-heavy tar ScoreCell_Score--scoreboard pl2')
    #live_scores = soup.find_all('div', class_='ScoreCell__Score h5 clr-gray-01 fw-heavy tar')
    game_type = soup.find_all('div', class_='ScoreboardScoreCell__Note clr-gray-04 n9 w-auto ml0')
    
    completed = soup.find_all('div', class_='ScoreCell__Time ScoreboardScoreCell__Time h9 clr-gray-01')
    progress = soup.find_all('div', class_='ScoreCell__Time ScoreboardScoreCell__Time h9 clr-negative')
    

    if len(teams) == 0:
        return 'No games found for this date'
    elif len(scores) == 0:
        game_time = soup.find_all('div', class_='ScoreCell__Time ScoreboardScoreCell__Time h9 clr-gray-03')
        network = soup.find_all('div', class_='ScoreCell__NetworkItem')
        embed = None 
        
        if game_type:
            matchup = game_type[0].text.split(',')
            series = matchup[1].strip() if len(matchup) == 2 else ''
            embed = discord.Embed(title=f"{completed[0].text}: {matchup[0]}: {teams[0].text} vs {teams[1].text}".strip(), description=f"{game_time[0].text} on {network[0].text}\n {series}".strip())
        else:
            embed = discord.Embed(title=f"{teams[0].text} vs {teams[1].text}", description=f"{game_time[0].text} on {network[0].text}")
        
        return embed
    else:
        if (len(completed) >= 1 and completed[0].text == 'Final'):
            embed = discord.Embed(title=f'{completed[0].text}: NBA Scores', color=0xff0000)
        elif (len(progress) >= 1 and progress[0].text != 'Final'):
            embed = discord.Embed(title=f'{progress[0].text}: NBA Scores', color=0x00ff00)
        for i in range(len(teams) // 2):
            if game_type:
                matchup = game_type[0].text.split(',')
                
                if int(scores[2 * i].text) > int(scores[2 * i + 1].text):
                    embed.add_field(name=f"{teams[2 * i].text} vs. {teams[2 * i + 1].text} ({matchup[0]})", value=f"**{teams[2 * i].text}: {scores[2 * i].text}**\n{teams[2 * i + 1].text}: {scores[2 * i + 1].text}\n{matchup[1].strip() if len(matchup) == 2 else ''}") 
                else:
                    embed.add_field(name=f"{teams[2 * i].text} vs. {teams[2 * i + 1].text} ({matchup[0]})", value=f"{teams[2 * i].text}: {scores[2 * i].text}\n**{teams[2 * i + 1].text}: {scores[2 * i + 1].text}**\n{matchup[1].strip() if len(matchup) == 2 else ''}")
            else:
                if int(scores[2 * i].text) > int(scores[2 * i + 1].text):
                    embed.add_field(name=f"{teams[2 * i].text} vs. {teams[2 * i + 1].text}", value=f"**{teams[2 * i].text}: {scores[2 * i].text}**\n{teams[2 * i + 1].text}: {scores[2 * i + 1].text}")
                else:
                    embed.add_field(name=f"{teams[2 * i].text} vs. {teams[2 * i + 1].text}", value=f"{teams[2 * i].text}: {scores[2 * i].text}\n**{teams[2 * i + 1].text}: {scores[2 * i + 1].text}**")                
        return embed