from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time
from bs4 import BeautifulSoup


# Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless") 
chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
chrome_options.add_argument("--log-level=3")  # Suppress logs

caps = DesiredCapabilities.CHROME
caps['goog:loggingPrefs'] = {'browser': 'OFF', 'performance': 'OFF'}
driver = webdriver.Chrome(options=chrome_options)


# URL to scrape
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}
curr_date = time.strftime('%Y%m%d')
url = f'https://www.espn.com/nba/scoreboard/_/date/{curr_date}'

def get_nba_score() -> str:
    # Scrape the page
    driver.get(url)
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    teams = soup.find_all('div', class_='ScoreCell__TeamName ScoreCell__TeamName--shortDisplayName truncate db')
    scores = soup.find_all('div', class_='ScoreCell__Score h4 clr-gray-01 fw-heavy tar ScoreCell_Score--scoreboard pl2')
    game_type = soup.find_all('div', class_='ScoreboardScoreCell__Note clr-gray-04 n9 w-auto ml0')

    if len(teams) == 0:
        return 'No games today'
    elif len(scores) == 0:
        game_time = soup.find_all('div', class_='ScoreCell__Time ScoreboardScoreCell__Time h9 clr-gray-03')
        network = soup.find_all('div', class_='ScoreCell__NetworkItem')
        
        if game_type:
            series = ''
            matchup = game_type[0].text.split(',')
            series += matchup[1].strip() if len(matchup) == 2 else ''
            return f"{matchup[0]}: {teams[0].text} vs {teams[1].text} at {game_time[0].text} on {network[0].text}\n{series}".strip()
        else:
            return f"{teams[0].text} vs {teams[1].text} today at {game_time[0].text} on {network[0].text}"
    else:
        nba_scores = '-----------------------\n'
        for i in range(len(teams) // 2):
            nba_scores += 'Game Score: '
            series = ''
            
            if game_type:
                matchup = game_type[0].text.split(',')
                nba_scores += f"{teams[2 * i].text} vs. {teams[2 * i + 1].text} ({matchup[0]})\n"
                series += matchup[1].strip() + '\n' if len(matchup) == 2 else ''
            else:
                nba_scores += f"{teams[2 * i].text} vs. {teams[2 * i + 1].text}\n"
                
            nba_scores += f"{teams[2 * i].text}: {scores[2 * i].text}\n"
            nba_scores += f"{teams[2 * i + 1].text}: {scores[2 * i + 1].text}\n"
            nba_scores += series
            nba_scores += '-----------------------\n'
        return nba_scores.strip()