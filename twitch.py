import asyncio
import os
import logging
from dotenv import load_dotenv
import time
from emoji import demojize
from datetime import datetime
import re
import pandas as pd

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
                print("Something went wrong with parsing")
            
    return pd.DataFrame().from_records(data)
        



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
    print("ehllo??")
    # Skipping welcome and join messages
    await reader.read(2048)
    try:
        await asyncio.wait_for(reader.read(2048), timeout = 5)

    except asyncio.TimeoutError: 
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
            print("here??")

            recv_msg = await asyncio.wait_for(reader.read(2048), timeout = 5)
            print("now here??")

        except asyncio.TimeoutError: 
            df = get_chat_dataframe('twitch_log.txt')
            a, b = df[df['channel'] == streamer.lower()].shape
            if (a < 1):
                print(df)
                return "Streamer not live and not in database"
            else: 
                print(df)
                return "Found in database"
        #recv_msg = await reader.read(2048)
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
            # time_logged = msg.split()[0].strip()

            # time_logged = datetime.strptime(time_logged, '%Y-%m-%d_%H:%M:%S')
            # username_message = msg.split('—')[1:]
            # username_message = '—'.join(username_message).strip()

            # print(f"user time: {time_logged}")
            # username, channel, message = re.search(':(.*)\!.*@.*\.tmi\.twitch\.tv PRIVMSG #(.*) :(.*)', username_message).groups()

            # print(f"Channel: {channel} \nUsername: {username} \nMessage: {message}")
    file.close()
    writer.close()
    await writer.wait_closed()
    df = get_chat_dataframe('twitch_log.txt')
    df['date'] = [d.date() for d in df['date/time']]
    df['time'] = [d.time() for d in df['date/time']]
    del df['date/time']
    df['channel'] = df['channel'].astype(str)
    df['username'] = df['username'].astype(str)
    df['message'] = df['message'].astype(str)

    print(df)
    return df.to_string()
