import os
import sys
# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
print(f"Parent diiir: {parent_dir}")
sys.path.append(parent_dir)

from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import User, PeerUser
from telethon.errors import FloodWaitError
import pandas as pd
import asyncio
import time 
import json

import re
import yaml

from helper_functions import find_repository_folder, find_files

config_path = os.path.join(find_repository_folder(), "config.yaml")
with open(config_path, 'r') as f:
    full_config = yaml.safe_load(f)

personal_parameters = full_config.get('personal_parameters', {})

api_id =                personal_parameters.get('TELEGRAM_API_ID')
api_hash =              personal_parameters.get('TELEGRAM_HASH_ID')
phone_number =          personal_parameters.get('PHONE_NUMBER')
session_name =          str(personal_parameters.get('SESSION_NAME'))

async def global_extract_dialog_info(client, messages):
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
                  
            # Connects the messages into full sentence even if the conversation is interrupted
            if text:
                  if last_message and sender == last_message[1]:
                        if last_message[0][-1] not in [".", "!", "?"]:
                            last_message[0] = last_message[0] + ","
                        
                        last_message[0] = " ".join([text, last_message[0]])
                  else:
                        if last_message:
                              extracted_dialog.append(last_message)
                        last_message = [text, sender, date]

      if last_message:
            extracted_dialog.append(last_message)
      
      return extracted_dialog

def local_extract_dialog_info(messages):
      extracted_dialog = []
      last_message=None

      for message in messages:
            try: 
                  text = message["text"] if type(message["text"]) == str else "" 
                  sender = message["from_id"] if message["from_id"] else message["from"] 
                  sender = re.sub(r'\D', '', sender)


                  date = message["date"]

            except Exception as e:
                  print(f"Exception: {e}")  
                                 
            if text:
                  if last_message and sender == last_message[1]:
                        if last_message[0][-1] not in [".", "!", "?"]:
                            last_message[0] = last_message[0] + ","
                        
                        text = text[0].lower() + text[1:]
                        last_message[0] = " ".join([text, last_message[0]])
                  else:
                        if last_message:
                              extracted_dialog.append(last_message)
                        last_message = [text, sender, date]
                    
      if last_message:
            extracted_dialog.append(last_message)
      
      return extracted_dialog

async def global_parse(
                     client: TelegramClient,
                     threshold: int =50, 
                     message_limit=None,
                     dialogs_limit: int = 100,
                     verbose=1,
                     checkpoints: bool = True
                     ):
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
                extracted_dialog = await global_extract_dialog_info(client, messages_info)
                filtered_dialogs = pd.concat([filtered_dialogs, pd.DataFrame(extracted_dialog, columns=["Message", "Sender", "Date"])])
                if verbose: 
                    total += 1
                    run_time = time.time() - start_time
                    print(f"Dialogs processed: {total}, left: {len(dialogs) - total}. Run time: {run_time:.2f} seconds") 
            if message_limit:
                print(len(filtered_dialogs))
                if len(filtered_dialogs) >= message_limit:
                    return filtered_dialogs
            if checkpoints:
                checkpoint = {"data": filtered_dialogs,
                               "last_iter": total}
                pd.to_pickle(checkpoint, "checkpoint.pkt")
        if os.path.exists("checkpoint.pkt"):
            os.remove("checkpoint.pkt")
        
        return filtered_dialogs
    
def local_parse(
            json_path: str,
            threshold: int=50, 
            message_limit=None,
            dialogs_limit: int=100,
            verbose: bool = 1,
            checkpoints: bool=True) -> pd.DataFrame:
      
      with open(json_path, "r") as f:
            data = json.load(f)
      
      # Will store message dialogs that will be processed
      filtered_dialogs = pd.DataFrame(columns=["DialogID", "Message", "Sender", "Date"])
      total_processed: int = 0

      def extract_messages(chat_info): 
            nonlocal filtered_dialogs, total_processed
            extracted_dialog = local_extract_dialog_info(chat_info["messages"])
            extracted_df = pd.DataFrame(extracted_dialog, columns=["Message", "Sender", "Date"])
            total_processed += 1
            extracted_df["DialogID"] = f"T_{total_processed}" # f"Telegram_1{i}"

            if verbose:
                  if total_processed % 20 == 0:
                        remaining_dialogs = len(data["chats"]["list"]) - total_processed
                        print(f"TELEGRAM: Processed {total_processed} dialogs. Left: {remaining_dialogs}")

            return extracted_df
      
      if verbose:
            print(f"TELEGRAM: Start processing. Total dialogs: {len(data['chats']['list'])}")

      for chat_info in data["chats"]["list"][:dialogs_limit]:
            if chat_info["type"] == "personal_chat" and len(chat_info["messages"]) > threshold:
                extracted_dialog = extract_messages(chat_info)
                filtered_dialogs = pd.concat([filtered_dialogs, extracted_dialog])

                if message_limit and len(filtered_dialogs) >= message_limit:
                    break

      return filtered_dialogs
    
async def main(parse_type: str, my_telegram_id = None, **kwargs):
    save_path = kwargs.get("save_path")
    del kwargs["save_path"]
    
    # Find our instagram dataset folder
    base_dir = find_repository_folder()
    json_path = find_files(base_dir, pattern="result\.json$")

    # Raise exception if no files found
    if len(json_path) == 0:
         raise ValueError("JSON file was not found. Please check the path and try again.") 
    json_path = json_path[0]

    # Parse types, local: from local file, global: through telegram api
    if parse_type == "local":
        if os.path.exists(json_path):
            data = local_parse(json_path=json_path, **kwargs)
        else: 
            raise ValueError(
                "JSON Path wasn't found. Please check the path and try again.")

    elif parse_type == "global":
            if os.path.exists(f"parsers/telegram/{session_name}.session-journal"):
                print(f"Session {session_name} exists. Please delete it and restart the script. Or change the session name in the script.")
                sys.exit()
            else:
                client = TelegramClient(session_name, api_id, api_hash)
                await client.start(phone_number)
                print(f"Connecting with {client.session}")
                data = await global_parse(client=client, **kwargs)
                my_telegram_id = (await client.get_me()).id
                client.disconnect()
    else: 
        raise ValueError("Invalid parse_type. Use 'local' or 'global'.")

    data["Sent_by_me"] = data["Sender"] == str(my_telegram_id) # Sent_by_me colummn for further processing

    if save_path:
        save_path = os.path.join(save_path , 'telegram_data.csv')
        if os.path.exists(save_path):
            if input("Telegram: File with the same name already exists. Do you want to overwrite it? (y/n)") == "y":
                data.to_csv(save_path, index=False)
            else:
                print("File not overwritten.")
        else: 
            data.to_csv(save_path, index=False)
    print("TELEGRAM: DONE")
        
    return data

if __name__ == "__main__":
    root_folder = find_repository_folder()
    config_path = os.path.join(root_folder, "config.yaml")
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    loading_parameters = full_config.get('loading_parameters', {})

    message_limit: int = loading_parameters.get('MESSAGE_LIMIT')
    dialogs_limit: int = loading_parameters.get('DIALOGS_LIMIT')
    verbose= loading_parameters.get('VERBOSE')                                           
    checkpoints: bool = loading_parameters.get('CHECKPOINTS')                   
    save_path = os.path.join(root_folder, loading_parameters.get('save_path'))
    threshold: int = loading_parameters.get('THRESHOLD')  

    kwargs = {
        "save_path": save_path,
        "message_limit": message_limit,
        "dialogs_limit": dialogs_limit,
        "verbose": verbose,
        "checkpoints": checkpoints,
        "threshold": threshold
    }

    data = asyncio.run(main(parse_type="local", **kwargs))