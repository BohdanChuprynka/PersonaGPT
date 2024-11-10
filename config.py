import os
import multiprocess as mp
from dotenv import load_dotenv
load_dotenv()

root_directory = os.path.dirname(os.path.abspath(__file__))

# Loading 
loading_parameters: dict = {
      "telegram": True,                                                                                                           # Whether parse telegram data
      "t_parse_type": "local",                                                                                                    # "local" or global" # Whether parse your messages through JSON Files that located locally (Fast way) or globally: (Via API) (takes 1 hour for ~20k messages) (Fill .env file)
      "t_local_json_path": os.path.abspath(os.path.join(root_directory, "parsers/telegram/result.json")),                        # If t_parse_type is "local", then fill it
      "t_global_save_path": os.path.abspath(os.path.join(root_directory, "parsers/telegram/result.csv")),                        # If t_parse_type is "global", then fill it
      "inbox_path": os.path.abspath(os.path.join(root_directory, "parsers/instagram/your_instagram_activity/messages/inbox")),    # Path to your instagram inbox
      "instagram": True,                                                                        # Whether parse instagram data
      "instagram_username": os.getenv('INSTAGRAM_USERNAME'),                                    # Your instagram username
      "discord": False,                                                                         # Whether parse discord data
      "discord_package_folder": os.path.abspath(os.path.join(root_directory, "parsers/discord/package")),        # Root folder that contains all the dialogs (Originally named "package")

      "message_limit": None,          # The maximum amount of messages to be processed total
      "dialogs_limit": None,          # The maximum amount of dialogs to be processed
      "verbose": 1,                   # The amount of output to be printed
      "checkpoints": True,            # To save data during parsing
      "threshold": 50,                # Dialogs with less than threshold messages will be dropped
      "save_csv": False,              # Drop the dialog if it has less or equal messages than the threshold
      "save_path": os.path.abspath(os.path.join(root_directory, "Datasets/")) # Where to save the data
      } 

# Processing

processing_parameters: dict = {
    "glove_path": os.path.abspath(os.path.join(root_directory, "Models/glove.6B.100d.txt")), # Synonym replacement model
    "chunks_path": os.path.abspath(os.path.join(root_directory, "data_chunks")),             # Path to the folder that contains the data chunks

    "censor_word": "CENSORED",                                  # The word that will be places instead of filtered sensitive word in filter_sensitive_words function.
    "context_size": 20,                                         # The amount of previous messages to include in the context column (20 by default)
    "num_chunks": 32,                                           # Number of chunks to split the dataset into (32 by default)
    "dataset_language": "uk",                                   # The native language of the dataset
    "back_translation_language": "en",                          # The language to be translated to and back from (en by default)
    "probs": [0.1, 0.3, 0.3, 0.3],                              # 0.1 for back-translation, 0.3 for shuffle, 0.3 for pop, 0.3 for swap. Lowered probabilities for back-translation because of low-resources
    "bool_synonym": True,                                       # Whether to perform synonym replacement together with back translation
    "synonym_percentage": 0.7,                                  # The amount of words to replace (70% default)
    "random_augmentation": True,                                # Whether to use random augmentation function on each sentence or not. (True by default)

    # Parallel processing
    "num_workers": mp.cpu_count()-2,                            # Parallel Computing: amount of cores to use in parallel computing 
    "memory_threshold": 2,                                      # Memory to leave available during augmentation. (2 by default) 
    "swap_processing": True,                                    # Swap memory in the process of augmentation. (True by default). Efficient in RAM. Instead of storing the whole dataset in RAM, it will swap it with disk.
    "delay": 10,                                                # Seconds to wait before continuing augmentation if memory_threshold is reached. (5 by default)
    "init_time": 5,                                             # Augmentation wrapper: Optimized in memory way of initializing the workers. Each workers will initialize for init_time after first worker. (5 default)
}
processing_kwargs: dict = {
      "augmentation_factor": 5,                                 # How many times to augment each question. (2 by default)
      "random_augmentation": True,                              # Whether to use random augmentation or not. (True by default)
      "samples": None}                                          # How much rows to process. (None/All by default)



# Training 
training_parameters: dict = {
      "DATA_PATH": os.path.join(root_directory, "Datasets/final_result.csv"),
      "MODEL_NAME": 'gpt2-medium',  
      "OUTPUT_DIR": os.path.join(root_directory, "Models/1.0v_PersonaGPT"),
      "MAX_LENGTH": 256,
      "BATCH_SIZE": 32,
      "EPOCHS": 3,
      "LEARNING_RATE": 5e-5,
      "WARMUP_RATIO": 5, # after (total_steps/3) steps stop warmup
      "SEED": 42
}
