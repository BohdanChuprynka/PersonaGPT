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
import json 
import time 
import openai

env_path = 'PersonaGPT/.env' 
load_dotenv(dotenv_path=env_path)


telegram: bool = True                                 # Whether parse telegram data
telegram_type = "locally"                             # "locally or globally" # Whether parse your messages through JSON Files that located locally (Fast way) or globally: (Via API) (takes 1 hour for ~20k messages)
# Requires openai for question generation
instagram: bool = False                               # Whether parse instagram data
inbox_path = "parsers/instagram/your_instagram_activity/messages/inbox"  # Path to your instagram inbox
instagram_username = os.getenv('INSTAGRAM_USERNAME')                 # Your instagram username
# Requires openai for question generation
discord: bool = False                                 # Whether parse discord data
discord_package_folder = "parsers/discord/package"    # Root folder that contains all the dialogs (Originally named "package")

message_limit: int = None                             # The maximum amount of messages to be processed total
dialogs_limit: int = None                             # The maximum amount of dialogs to be processed
verbose=1                                             # The amount of output to be printed
checkpoints: bool = True                              # To save data during parsing
threshold: int = 50                                   # Drop the dialog if it has less or equal messages than the threshold


def optimize_messages(messages):
      """
      Function which uses a set of tuning algorithms to meet the criteria of optimized data for future models.
      """

      # TODO: Include only messages in ukrainian language


      # TODO: Put todos below in order of priority 
      # For each of the points below, if true: add one, if false: minus one
      # TODO: Add detection system for context and response:

      # TODO: If the message contains question mark in the end of the message, it is a context
      # TODO: The first message of the new day is probably a context.
      # TODO: If there are few messages in a row from user, concatenate them into one message.
      # TODO: If there is a significant time gap (e.g., several hours) between messages, the first message after the gap might be a context.
      # TODO: Look for specific keywords or phrases that typically indicate a context (e.g., "What do you think about...", "Can you explain...", "Why is...").
      # TODO: If a message is a direct reply to a previous context message, it is likely a response.
      # TODO: Short messages that directly follow a context are likely responses.
      # TODO: If the same user repeatedly sends messages ending with question marks or messages at the start of the day, those are likely contexts.
       
def generate_context(response): # GPT Required 
      """
      Uses AI to generate context based on response
      """ 
      #openai.api_key = os.getenv('OPENAI_API_KEY')

      prompt = f'Classify the following message as either "context" or "response":\n\n"{message}"\n\nAnswer with one word only.'
      
      response = openai.Completion.create(
            engine="GPT-4",  # Choose a suitable engine
            prompt=prompt,
            max_tokens=10,
            n=1,
            stop=None,
            temperature=0.5,
      )
      
      return response.choices[0].text.strip()

async def main(telegram = telegram,
               instagram = instagram,
               discord = discord,
               discord_path = discord_package_folder,
               instagram_path = inbox_path,
               **kwargs): 
      
      if telegram:
            telegram_df = await telegram_parse.main(parse_type=telegram_type,**kwargs,)
      if instagram:
            instagram_df = instagram_parse.main(inbox_path=inbox_path, instagram_username=instagram_username, **kwargs)
      if discord:
            discord_df = discord_parse.main(path=discord_path, **kwargs)

if __name__ == "__main__": 

      kwargs = {
            "message_limit": message_limit,
            "dialogs_limit": dialogs_limit,
            "verbose": verbose,
            "checkpoints": checkpoints,
            "threshold": threshold
      }

      loop = asyncio.get_event_loop()
      loop.run_until_complete(main(**kwargs))
      