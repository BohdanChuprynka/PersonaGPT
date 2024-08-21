# Python file specifically for DataCollector.py. '
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import User, PeerUser
from telethon.errors import FloodWaitError
import pandas as pd
import asyncio
import time 
import openai
from dotenv import load_dotenv
import json
import os
import sys

dotenv_path = ".env"
load_dotenv(dotenv_path=dotenv_path)

api_id = os.getenv('TELEGRAM_API_ID')
api_hash = os.getenv('TELEGRAM_HASH_ID')
phone_number = os.getenv('PHONE_NUMBER')
session_name = "telegram_parser"
client = TelegramClient(session_name, api_id, api_hash)

async def extract_message_info(messages):
      extracted_dialog = []
      last_message=None

      for message in messages:
            try: 
                  text = message.message.strip() if message.message else ""
                  sender = message.from_id if message.from_id else (await client.get_entity(message.peer_id)).id
                  sender = sender.user_id if isinstance(sender, PeerUser) else sender # Deletes PeerUser classes and keeps only int id

                  date = message.date 
            except FloodWaitError as e:
                  print(f"FloodWaitError: sleeping for {e.seconds} seconds.")
                  await asyncio.sleep(e.seconds)
                  continue 
                  
            if text:
                  if last_message and sender == last_message[1]:
                        last_message[0] = " ".join([last_message[0], text])
                  else:
                        if last_message:
                              extracted_dialog.append(last_message)
                        last_message = [text, sender, date]

      if last_message:
            extracted_dialog.append(last_message)
      
      return extracted_dialog

async def parse_data(threshold: int =50, 
                     message_limit=None,
                     dialogs_limit: int = 100,
                     verbose=1,
                     checkpoints: bool = True):
    """
    Parses all the messages in the profile.
    
    Args:
        threshold: int
            The minimum amount of messages in a dialog to be processed.
        message_limit: int
            The maximum amount of messages to be processed in a dialog.
        dialogs_limit: int
            The maximum amount of dialogs to be processed.
        verbose: int
            The amount of output to be printed.
        top_chats_first: bool
            Whether to process chats with most messages first.

    Returns:
        pd.DataFrame:
            prepared DataFrame with columns ["Message", "Sender", "Date"]
    """
    async with client:
        dialogs = await client.get_dialogs()
        dialogs = [dialog for dialog in dialogs if isinstance(dialog.entity, User)]
        dialogs = [dialog for dialog in dialogs if not dialog.entity.bot]
        my_telegram_id = int((await client.get_me()).id)
        dialogs = [dialog for dialog in dialogs if dialog.entity.id != my_telegram_id]
        dialogs = dialogs[:dialogs_limit]
        filtered_dialogs = pd.DataFrame(columns=["Message", "Sender","Date"])

        if verbose: 
            total = 0
            print(f"Total dialogs: {len(dialogs)}")

        # Check for checkpoint
        if checkpoints:
            if os.path.exists("checkpoint.pkt"):
                checkpoint = pd.read_pickle("checkpoint.pkt")
                filtered_dialogs = checkpoint["data"]
                total = checkpoint["last_iter"]
                dialogs = dialogs[total-1:]
                print(f"Resuming from checkpoint. Dialogs left: {len(dialogs)}.")

        # Main loop of the function
        for dialog in dialogs[:dialogs_limit]:  
            start_time = time.time() if verbose else None
            messages_info = []
            async for message in client.iter_messages(dialog.entity, limit=message_limit, wait_time=10):
                messages_info.append(message)

            total_messages = len(messages_info)
            if total_messages > threshold:
                extracted_dialog = await extract_message_info(messages_info)
                filtered_dialogs = pd.concat([filtered_dialogs, pd.DataFrame(extracted_dialog, columns=["Message", "Sender", "Date"])])
                if verbose: 
                    total += 1
                    run_time = time.time() - start_time
                    print(f"Dialogs processed: {total}, left: {len(dialogs) - total}. Run time: {run_time:.2f} seconds") 
            if checkpoints:
                checkpoint = {"data": filtered_dialogs,
                               "last_iter": total}
                pd.to_pickle(checkpoint, "checkpoint.pkt")
        if os.path.exists("checkpoint.pkt"):
            os.remove("checkpoint.pkt")
        
        return filtered_dialogs
    
async def main(message_limit: int = None, dialogs_limit: int = None, verbose=1, checkpoints: bool = True, threshold: int = 50):
    if os.path.exists(f"parsers/telegram/{session_name}.session-journal"):
        print(f"Session {session_name} exists. Please delete it and restart the script. Or change the session name in the script.")
        sys.exit()
    else:
        await client.start(phone_number)
        print(f"Connecting with {client.session}")
        data = await parse_data(message_limit=message_limit, dialogs_limit=dialogs_limit, verbose=verbose, checkpoints=checkpoints, threshold=threshold)
        data = pd.DataFrame(data, columns=["Message", "Sender", "Date"])
        my_telegram_id = (await client.get_me()).id
        data["Sent_by_me"] = int(my_telegram_id) == data["Sender"]
        return data
        client.disconnect()

        if os.path.exists(r"parsers/telegram/full_telegram_data.csv"):
            print("File with the same name already exists. Do you want to overwrite it? (y/n)")
            if input() == "y":
                  data.to_csv(r'full_telegram_data.csv', index=False)
            else:
                  print("File not overwritten.")
                  sys.close()
        print("TELEGRAM: DONE")

