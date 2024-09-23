import json 
import os
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

env_path = "PersonaGPT/.env"
load_dotenv(dotenv_path=env_path)

# Function to extract information from messages.json
def extract_message_info(json_file_path, message_limit: int = None):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        dialog = json.load(file)
        for row in dialog:
            timestamp = row.get('Timestamp')
            content = row.get('Contents')
            return [content, timestamp]
        

def main(path: str = None, save_csv: bool = True, **kwargs):
      message_limit = kwargs.get("message_limit")
      dialogs_limit = kwargs.get("dialogs_limit")
      verbose = kwargs.get("verbose")

      if not os.path.exists(path):
            print(f"Discord Directory '{path}' does not exist.")
            exit()

      data = []
      limit = False
      if verbose: 
            print(f"Discord data is processed from Path: {path}")
            
      for root, dirs, files in os.walk(path):
            if not limit:
                  for file in files[:dialogs_limit]:
                        if file == 'messages.json':
                              json_file_path = os.path.join(root, file)
                              processed_row = extract_message_info(json_file_path, message_limit)
                              data.append(processed_row)
                              
                              if message_limit and len(data) >= message_limit:
                                    limit= True
                                    break
      
      try: 
            data = pd.DataFrame(data, columns=['Message', 'Date'])
      except Exception as e:
            print(f"Fixing exception: {e}")
            data = [row for row in data if row is not None]
            data = pd.DataFrame(data, columns=['Message', 'Date'])
      
      if save_csv:
            folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets'))
            save_path = os.path.join(folder_path, 'discord_data.csv')
            if os.path.exists(save_path):
                  print("File with the same name already exists. Do you want to overwrite it? (y/n)")
                  if input() == "y":
                        data.to_csv(save_path, index=False)
                        print("File overwritten.")
                  else:
                        print("File not overwritten.")
            else: 
                  data.to_csv(save_path, index=False)
                  print("File created.")
                  
      print("Discord: DONE ")
      return data 
      
if __name__ == "__main__":
      
      message_limit: int = None                             # The maximum amount of messages to be processed total
      dialogs_limit: int = None                             # The maximum amount of dialogs to be processed
      verbose=1                                             # The amount of output to be printed
      checkpoints: bool = True                              # To save data during parsing
      threshold: int = 50                                   # Drop the dialog if it has less or equal messages than the threshold
      save_csv: bool = True         
      path = str(os.getenv('DISCORD_PATH'))

      kwargs = {
            "save_csv": save_csv,
            "message_limit": message_limit,
            "dialogs_limit": dialogs_limit,
            "verbose": verbose,
            "checkpoints": checkpoints,
            "threshold": threshold
      }

      main(path=path, save_csv=True)