import json 
import os
import numpy as np
import pandas as pd
import yaml

from helper_functions import find_repository_folder, find_dirs

def decode_utf8(encoded_str):
      # Decoding the string
      try: 
            decoded_str = str(encoded_str)
            decoded_str = encoded_str.encode('latin1').decode('utf-8')
            return decoded_str
      except AttributeError:
            pass 

def extract_dialog(json_file_path, message_limit: int = None, dialogs_limit: int = None, verbose=1, checkpoints: bool = True, threshold: int = 50): 
      with open(json_file_path, 'r', errors='replace') as file:
            dialog = json.load(file)
            if threshold and len(dialog["messages"]) < threshold:
                  return
            
            last_message = None
            extracted_dialog = []
            for message_data in dialog["messages"]:
                  encoded_message = message_data["content"] if "content" in message_data else np.nan
                  message = decode_utf8(encoded_message)

                  sender = message_data["sender_name"]

                  timestamp = message_data["timestamp_ms"]
                  timestamp = pd.to_datetime(timestamp, unit='ms')
                  
                  if message:
                        if last_message and sender == last_message[1]:
                              if last_message[0][-1] not in [".", "!", "?"]:
                                    last_message[0] = last_message[0] + ","
                              
                              last_message[0] = " ".join([message.lower(), last_message[0]])
                        else:
                              if last_message:
                                    extracted_dialog.append(last_message)
                              last_message = [message, sender, timestamp]
            
            return extracted_dialog


def main(instagram_username: str, **kwargs):
      verbose = kwargs.get("verbose")
      save_path = kwargs.get("save_path")
      message_limit = kwargs.get("message_limit")
      dialogs_limit = kwargs.get("dialogs_limit")
      threshold = kwargs.get("threshold")
      del kwargs["save_path"]


      base_dir = find_repository_folder()
      inbox_path = find_dirs(base_dir, pattern="inbox$")

      # Raise exception if no files found
      if len(inbox_path) == 0:
            raise ValueError("Instagram: Inbox folder was not found. Please check the path and try again.") 
      inbox_path = inbox_path[0]


      df = pd.DataFrame(columns=["DialogID", 'Message', 'Sender', 'Date'])   
      dialogs_paths = []

      for root, dirs, files in os.walk(inbox_path):
            for file in files:
                  if file == 'message_1.json':
                        json_file_path = os.path.join(root, file)
                        dialogs_paths.append(json_file_path) 

      if verbose: 
            print(f"Instagram Data is processing. Total dialogs: {len(dialogs_paths)}")
      for i, path in enumerate(dialogs_paths[:dialogs_limit]):
            
            data = extract_dialog(json_file_path=path, 
                                  message_limit=message_limit,
                                  dialogs_limit=dialogs_limit,
                                  verbose=1,
                                  threshold=threshold)
            data = pd.DataFrame(data, columns=['Message', 'Sender', 'Date'])
            data["DialogID"] = f"I_{i}" # f"Instagram_{i}"
            df = pd.concat([df, data])
            
            if message_limit and len(df) >= message_limit:
                  break
            
      df["Sent_by_me"] = df["Sender"] == str(instagram_username)
      
      if save_path:
            save_path = os.path.join(save_path, 'instagram_data.csv')
            print(f"Saving data to {save_path}")
            if os.path.exists(save_path):
                  if input("Instagram: File with the same name already exists. Do you want to overwrite it? (y/n)") == "y":
                        df.to_csv(save_path, index=False)
                        print("Instagram: File overwritten.")
                  else:
                        print("File not overwritten.")
            else: 
                  df.to_csv(save_path, index=False)


      print("Instagram: DONE")
      return df


if __name__ == "__main__":
      # Parameters      
      root_path = find_repository_folder()
      message_limit: int = None                             # The maximum amount of messages to be processed total
      dialogs_limit: int = None                             # The maximum amount of dialogs to be processed
      verbose=1                                             # The amount of output to be printed
      checkpoints: bool = True                              # To save data during parsing
      threshold: int = 50                                   # Drop the dialog if it has less or equal messages than the threshold
      save_dir = os.getcwd(z)

      kwargs = {
            "save_path": save_dir,
            "message_limit": message_limit,
            "dialogs_limit": dialogs_limit,
            "verbose": verbose,
            "checkpoints": checkpoints,
            "threshold": threshold
      }

      config_path = os.path.join(os.getcwd(), "config.yaml")
      with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)

      personal_parameters = full_config.get('personal_parameters', {})

      instagram_username = personal_parameters.get('INSTAGRAM_USERNAME')

      if not instagram_username: 
            raise ValueError("Instagram username is not set in the .config file.")

      main(instagram_username=instagram_username, **kwargs)