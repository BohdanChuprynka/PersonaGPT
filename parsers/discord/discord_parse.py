# Function to extract information from messages.json
def extract_message_info(json_file_path, message_limit: int = None):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        dialog = json.load(file)
        for row in dialog:
            timestamp = row.get('Timestamp')
            content = row.get('Contents')
            return [content, timestamp]
        

def main(message_limit: int = None, dialogs_limit: int = None, verbose=1, checkpoints: bool = True, threshold: int = 50, path: str = None, save_csv: bool = True):
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
            if os.path.exists(r"parsers/discord/discord_data.csv"):
                  print("File with the same name already exists. Do you want to overwrite it? (y/n)")
                  if input() == "y":
                        print("File overwritten.")
                        data.to_csv(r'parsers/discord/discord_data.csv', index=False)
                  else:
                        print("File not overwritten.")
            else: 
                  data.to_csv(r'parsers/discord/discord_data.csv', index=False)
                  print("File created.")
                  
      print("Discord: DONE ")
      return data 
      
if __name__ == "__main__":
      import json 
      import os
      import numpy as np
      import pandas as pd

      path = "parsers/discord/package/messages"
      main(path=path, save_csv=True)