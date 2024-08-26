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
                              last_message[0] = last_message[0].lower()
                              last_message[0] = " ".join([last_message[0], message])
                        else:
                              if last_message:
                                    extracted_dialog.append(last_message)
                              last_message = [message, sender, timestamp]
            
            return extracted_dialog


def main(inbox_path: str, instagram_username: str, message_limit: int = None, dialogs_limit: int = None, verbose=1, threshold: int = 50):
      if not os.path.exists(inbox_path):
                  raise Exception(f"Directory '{inbox_path}' for instagram folder wasn't found.\nTry to change the path to your_instagram_activity -> messages -> inbox.")


      df = pd.DataFrame(columns=['Message', 'Sender', 'Date'])   
      dialogs_path = []

      for root, dirs, files in os.walk(inbox_path):
            for file in files:
                  if file == 'message_1.json':
                        json_file_path = os.path.join(root, file)
                        dialogs_path.append(json_file_path) 

      if verbose: 
            print(f"Instagram Data is processing. Total dialogs: {len(dialogs_path)}")
      for path in dialogs_path[:dialogs_limit]:
            
            data = extract_dialog(json_file_path=path, 
                                  message_limit=message_limit,
                                  dialogs_limit=dialogs_limit,
                                  verbose=1,
                                  threshold=threshold)
            data = pd.DataFrame(data, columns=['Message', 'Sender', 'Date'])
            df = pd.concat([df, data])
            
            if message_limit and len(df) >= message_limit:
                  break
            
      df["Sent_by_me"] = df["Sender"] == str(instagram_username)
      print("Instagram: DONE")
      return df


if __name__ == "__main__":
      import json 
      import os
      import numpy as np
      import pandas as pd
      from dotenv import load_dotenv

      env_path = 'PersonaGPT/.env'
      load_dotenv(dotenv_path=env_path)
      inbox_path = str(os.getenv('INBOX_PATH'))
      instagram_username = os.getenv('INSTAGRAM_USERNAME')

      main(inbox_path=inbox_path, instagram_username=instagram_username)