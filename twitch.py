import asyncio
import os
import logging
from dotenv import load_dotenv
import time
from emoji import demojize
from datetime import datetime
import re
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import discord

#credit: https://www.learndatasci.com/tutorials/how-stream-text-data-twitch-sockets-python/

def get_chat_dataframe(file):
    data = []

    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().split('\n')
        for line in lines:
            try:
                time_logged = line.split('--')[0].strip()
                time_logged = datetime.strptime(time_logged, '%Y-%m-%d_%H:%M:%S')

                username_message = line.split('--')[1:]
                username_message = '--'.join(username_message).strip()

                username, channel, message = re.search(
                    ':(.*)\!.*@.*\.tmi\.twitch\.tv PRIVMSG #(.*) :(.*)', username_message
                ).groups()

                d = {
                    'date/time': time_logged,
                    'channel': channel,
                    'username': username,
                    'message': message
                }

                data.append(d)
            
            except Exception:
                pass
            
    return pd.DataFrame().from_records(data)
        
# credit: https://www.datacamp.com/tutorial/text-analytics-beginners-nltk
def preprocess_text(text):
    nltk.download('all')
    #stopwords.words('english').append()
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens if (re.sub(r'[^\w\s]', '', token) and (token not in stopwords.words('english')))]
    lemmatizer = WordNetLemmatizer()
    lemm_tokens = [lemmatizer.lemmatize(token) for token in cleaned_tokens]
    return ' '.join(lemm_tokens)

def get_pos_words(df: pd.DataFrame):
    pos_lines = list(df[df.label == 1].message)
    tokens = []
    for line in pos_lines:
        toks = word_tokenize(line)
        tokens.extend(toks)
    pos_freq = nltk.FreqDist(tokens)
    words = pos_freq.most_common(10)
    ans = "Top positive words:\n"
    for word in words:
        ans += f"- {word[0]}\n"
    return ans
def get_neg_words(df: pd.DataFrame):
    neg_lines = list(df[df.label == -1].message)
    tokens = []
    for line in neg_lines:
        toks = word_tokenize(line)
        tokens.extend(toks)
    neg_freq = nltk.FreqDist(tokens)
    words = neg_freq.most_common(10)
    ans = "Top negative words:\n"
    for word in words:
        ans += f"- {word[0]}\n"
    return ans

def get_sentiment (sent, text) :
    #nltk.download('all')
    sia = SIA()
    scores = sia.polarity_scores(text)

    if sent == "pos":
        return scores['pos']
    elif sent == "neu" :
        return scores['neu']
    elif sent =="neg" :
        return scores['neg']
    else :
        return scores['compound']

def data_analysis () :
    df = get_chat_dataframe('twitch_log.txt')
    #data cleaning
    df['date'] = [d.date() for d in df['date/time']]
    df['time'] = [d.time() for d in df['date/time']]
    del df['date/time']
    df['channel'] = df['channel'].astype(str)
    df['username'] = df['username'].astype(str)
    df['message'] = df['message'].astype(str)
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day'] = pd.DatetimeIndex(df['date']).day

    df['message'] = df['message'].apply(preprocess_text)
    df['pos'] = [get_sentiment("pos", mes) for mes in df['message']]
    df['neu'] = [get_sentiment("neu", mes) for mes in df['message']]
    df['neg'] = [get_sentiment("neg", mes) for mes in df['message']]
    df['tot'] = [get_sentiment("tot", mes) for mes in df['message']]
    df['label'] = 0
    df.loc[df['tot'] > 0.2, 'label'] = 1
    df.loc[df['tot'] < -0.2, 'label'] = -1
    return df


async def get_message(streamer: str) -> str:
    load_dotenv()
    oauth = os.getenv('TWITCH_OAUTH')
    server = 'irc.chat.twitch.tv'
    port = 6667
    nickname = 'PabloBot'
    channel = "#"+streamer

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s — %(message)s',
                        datefmt='%Y-%m-%d_%H:%M:%S',
                        handlers=[logging.FileHandler('chat.log', encoding='utf-8')])
    file = open("twitch_log.txt", "a")

    reader, writer = await asyncio.open_connection(server, port)
    
    writer.write(f"PASS {oauth}\n".encode('utf-8'))
    await writer.drain()
    writer.write(f"NICK {nickname}\n".encode('utf-8'))
    await writer.drain()
    writer.write(f"JOIN {channel}\n".encode('utf-8'))
    await writer.drain()
    # Skipping welcome and join messages
    isLive = True
    await reader.read(2048)
    try:
        await asyncio.wait_for(reader.read(2048), timeout = 5)

    except asyncio.TimeoutError: 
        isLive = False
        df = get_chat_dataframe('twitch_log.txt')
        a,b = df.loc[df['channel'] == streamer.lower()].shape
        if (a < 1):
            return "Streamer doesn't exist."
        else: 
            print(df)
            return "Found in database"

    end_time = time.time() + 10  # run for 10 seconds
    response = ""

    while time.time() < end_time:
        try:
            recv_msg = await asyncio.wait_for(reader.read(2048), timeout = 5)

        except asyncio.TimeoutError: 
            isLive = False
            df = get_chat_dataframe('twitch_log.txt')
            a, b = df[df['channel'] == streamer.lower()].shape
            if (a < 1):
                print(df)
                return "Streamer not live and not in database"
            else: 
                print(df)
                print("Found in database")
        #recv_msg = await reader.read(2048)
        if isLive:
            recv_msg = recv_msg.decode('utf-8')
            
            if recv_msg.startswith('PING'):
                writer.write("PONG\n".encode('utf-8'))
                await writer.drain()
            elif len(recv_msg) > 0:

                timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
                msg = f"{timestamp} -- {demojize(recv_msg)}"
                file.write(msg)
                file.flush()
                logging.info(msg)
                response += msg
        file.close()
        writer.close()
        await writer.wait_closed()
    df = data_analysis()
    print(df)
    df2 = df.loc[df['channel'] == streamer.lower()]
    print("take two")
    print(df2)
    print("haha")

    df_user = df2.groupby(['username']).count()
    #df_user : pd.DataFrame = df_user.loc[len(df['username']) < 50]
    print(f"yoinnk {df_user.shape}")
    print(df_user)
    df_user_top : pd.DataFrame = df_user.sort_values(by='channel', ascending=False).head(10)
    print(df_user_top)
    df_user_top.plot.bar(y='message',legend=False)
    plt.xticks(rotation=55, ha='right')

    # Setting plot labels
    plt.xlabel('Username')
    plt.ylabel('Message Count')
    plt.title('Top Users by Message Count')
    plt.savefig("plot.png", bbox_inches='tight', pad_inches=0.4)

    return discord.Embed(title=f"Streamer: {streamer}", description=f"users stored: {df_user.shape[0]}\nAvg total sentiment (out of 1): {df2['tot'].mean():.2f}\nAvg pos: {df2['pos'].mean():.2f}\nAvg neutral: {df2['neu'].mean():.2f}\nAvg neg: {df2['neg'].mean():.2f}\n{get_pos_words(df2)}\n{get_neg_words(df2)}", color= 666531)
