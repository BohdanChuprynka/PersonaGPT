from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import User, PeerUser
from telethon.errors import FloodWaitError
from telethon import TelegramClient
from dotenv import load_dotenv 

import parsers.telegram.telegram_parse as telegram_parse
import parsers.discord.discord_parse as discord_parse
import parsers.instagram.instagram_parse as instagram_parse
import pandas as pd
import numpy as np 
import asyncio
import os 
import re
import json 
import time 
import openai

env_path = 'PersonaGPT/.env' 
load_dotenv(dotenv_path=env_path)


telegram: bool = True                                 # Whether parse telegram data
t_parse_type = "local"                                # "local" or global" # Whether parse your messages through JSON Files that located locally (Fast way) or globally: (Via API) (takes 1 hour for ~20k messages) (Fill .env file)
json_path = "/Users/bohdan/Documents/Programming/Projects/VSCode/AI-DataScience/PersonaGPT/parsers/telegram/result.json" # If t_parse_type is "local", then fill it
telegram_save_path = r"/Users/bohdan/Documents/Programming/Projects/VSCode/AI-DataScience/PersonaGPT/parsers/telegram/result.csv"  # If t_parse_type is "global", then fill it
# Requires openai for question generation
instagram: bool = True                               # Whether parse instagram data
inbox_path = "parsers/instagram/your_instagram_activity/messages/inbox"  # Path to your instagram inbox
instagram_username = os.getenv('INSTAGRAM_USERNAME')                 # Your instagram username
# Requires openai for question generation
discord: bool = True                                 # Whether parse discord data
discord_package_folder = "parsers/discord/package"    # Root folder that contains all the dialogs (Originally named "package")

message_limit: int = None                             # The maximum amount of messages to be processed total
dialogs_limit: int = None                             # The maximum amount of dialogs to be processed
verbose=1                                             # The amount of output to be printed
checkpoints: bool = True                              # To save data during parsing
threshold: int = 50      
save_csv: bool = True                                 # Drop the dialog if it has less or equal messages than the threshold
       
async def main(telegram = telegram,
               instagram = instagram,
               discord = discord,
               discord_path = discord_package_folder,
               **kwargs): 

      if telegram:
            telegram_df = await telegram_parse.main(parse_type=t_parse_type,json_path=json_path, save_path=telegram_save_path, **kwargs,)
      if instagram:
            instagram_df = instagram_parse.main(inbox_path=inbox_path, instagram_username=instagram_username, **kwargs)
      if discord:
            discord_df = discord_parse.main(path=discord_path, **kwargs)

      print("Done!")

if __name__ == "__main__": 

      kwargs = {
            "save_csv": save_csv,             
            "message_limit": message_limit,
            "dialogs_limit": dialogs_limit,
            "verbose": verbose,
            "checkpoints": checkpoints,
            "threshold": threshold
      }

      loop = asyncio.get_event_loop()
      loop.run_until_complete(main(**kwargs))
      