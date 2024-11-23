from config import processing_parameters
from config import processing_kwargs as kwargs

import numpy as np
import pandas as pd
from googletrans import Translator
import gensim.downloader as api
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf
import psutil
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from functools import cache
from nlpaug.util import Action
from functools import partial

import os
import time
from tqdm import tqdm
import random

# Global Variables
import multiprocess as mp              # NOT multiprocessing to avoid __main__ improtable problem by the children 
import requests
from dotenv import load_dotenv
env_path = "PersonaGPT/.env"
load_dotenv(dotenv_path=env_path)

# Initialize parameters from config.py
if processing_parameters:
    locals().update(processing_parameters)
else:
    Warning("Custom data cleaning parameters are not used. Setting up default parameters.")
    root_directory = os.path.dirname(os.getcwd())
    glove_path = os.path.join(root_directory, "Models/glove.6B.100d.txt") # Synonym replacement model
    chunks_path = os.path.join(root_directory, "data_chunks")             # Path to the folder that contains the data chunks

    censor_word: str = "CENSORED"          # The word that will be places instead of filtered sensitive word in filter_sensitive_words function.
    context_size: int = 20                 # The amount of previous messages to include in the context column (20 by default)
    num_chunks: int = 32                   # Number of chunks to split the dataset into (32 by default)
    dataset_language: str = "uk"           # The native language of the dataset
    back_translation_language: str = "en"  # The language to be translated to and back from (en by default)
    probs = [0.1, 0.3, 0.3, 0.3]           # 0.1 for back-translation, 0.3 for shuffle, 0.3 for pop, 0.3 for swap. Lowered probabilities for back-translation because of low-resources
    bool_synonym: bool = True              # Whether to perform synonym replacement together with back translation
    synonym_percentage: int = 0.7          # The amount of words to replace (70% default)
    random_augmentation: bool = True       # Whether to use random augmentation function on each sentence or not. (True by default)
    kwargs: dict = {
        "augmentation_factor": 5,          # How many times to augment each question. (2 by default)
        "random_augmentation": True,       # Whether to use random augmentation or not. (True by default)
        "samples": None}                   # How much rows to process. (None by default)

    # Parallel processing
    num_workers: int = mp.cpu_count()-2    # Parallel Computing: amount of cores to use in parallel computing 
    memory_threshold: float = 2            # Memory to leave available during augmentation. (2 by default) 
    swap_processing: bool = True           # Swap memory in the process of augmentation. (True by default). Efficient in RAM. Instead of storing the whole dataset in RAM, it will swap it with disk.
    delay: int = 10                        # Seconds to wait before continuing augmentation if memory_threshold is reached. (5 by default)
    init_time: int = 5                     # Augmentation wrapper: Optimized in memory way of initializing the workers. Each workers will initialize for init_time after first worker. (5 default)

keys_to_filter = os.getenv('KEYS_TO_FILTER').split(',')  # list of sensitive words
concatenated_path = "Datasets/concatenated.csv"           
english_topwords = set(stopwords.words('english'))       # English stopwords

LANG_CODES = {
    'afrikaans': 'af',
    'albanian': 'sq',
    'amharic': 'am',
    'arabic': 'ar',
    'armenian': 'hy',
    'azerbaijani': 'az',
    'basque': 'eu',
    'belarusian': 'be',
    'bengali': 'bn',
    'bosnian': 'bs',
    'bulgarian': 'bg',
    'catalan': 'ca',
    'cebuano': 'ceb',
    'chichewa': 'ny',
    'chinese (simplified)': 'zh-cn',
    'chinese (traditional)': 'zh-tw',
    'corsican': 'co',
    'croatian': 'hr',
    'czech': 'cs',
    'danish': 'da',
    'dutch': 'nl',
    'english': 'en',
    'esperanto': 'eo',
    'estonian': 'et',
    'filipino': 'tl',
    'finnish': 'fi',
    'french': 'fr',
    'frisian': 'fy',
    'galician': 'gl',
    'georgian': 'ka',
    'german': 'de',
    'greek': 'el',
    'gujarati': 'gu',
    'haitian creole': 'ht',
    'hausa': 'ha',
    'hawaiian': 'haw',
    'hebrew': 'he',
    'hindi': 'hi',
    'hmong': 'hmn',
    'hungarian': 'hu',
    'icelandic': 'is',
    'igbo': 'ig',
    'indonesian': 'id',
    'irish': 'ga',
    'italian': 'it',
    'japanese': 'ja',
    'javanese': 'jw',
    'kannada': 'kn',
    'kazakh': 'kk',
    'khmer': 'km',
    'korean': 'ko',
    'kurdish (kurmanji)': 'ku',
    'kyrgyz': 'ky',
    'lao': 'lo',
    'latin': 'la',
    'latvian': 'lv',
    'lithuanian': 'lt',
    'luxembourgish': 'lb',
    'macedonian': 'mk',
    'malagasy': 'mg',
    'malay': 'ms',
    'malayalam': 'ml',
    'maltese': 'mt',
    'maori': 'mi',
    'marathi': 'mr',
    'mongolian': 'mn',
    'myanmar (burmese)': 'my',
    'nepali': 'ne',
    'norwegian': 'no',
    'odia': 'or',
    'pashto': 'ps',
    'persian': 'fa',
    'polish': 'pl',
    'portuguese': 'pt',
    'punjabi': 'pa',
    'romanian': 'ro',
    'russian': 'ru',
    'samoan': 'sm',
    'scots gaelic': 'gd',
    'serbian': 'sr',
    'sesotho': 'st',
    'shona': 'sn',
    'sindhi': 'sd',
    'sinhala': 'si',
    'slovak': 'sk',
    'slovenian': 'sl',
    'somali': 'so',
    'spanish': 'es',
    'sundanese': 'su',
    'swahili': 'sw',
    'swedish': 'sv',
    'tajik': 'tg',
    'tamil': 'ta',
    'telugu': 'te',
    'thai': 'th',
    'turkish': 'tr',
    'ukrainian': 'uk',
    'urdu': 'ur',
    'uyghur': 'ug',
    'uzbek': 'uz',
    'vietnamese': 'vi',
    'welsh': 'cy',
    'xhosa': 'xh',
    'yiddish': 'yi',
    'yoruba': 'yo',
    'zulu': 'zu'}

LANGUAGES = {value:key for key, value in LANG_CODES.items()}

# Get ukrainian stopwords
if not os.path.exists("Datasets/stopwords_ua_set.txt"):
      directory = 'Datasets/'
      url = "https://raw.githubusercontent.com/skupriienko/Ukrainian-Stopwords/refs/heads/master/stopwords_ua_set.txt"
      os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist

      # Filename to save the file as
      filename = os.path.join(directory, 'stopwords_ua_set.txt')
      # Download and save the file
      response = requests.get(url)
      with open(filename, 'wb') as f:
            f.write(response.content)
      print(f"File saved as {filename}")

with open('Datasets/stopwords_ua_set.txt', 'r') as file:
    ukrainian_stop_words = file.read().splitlines()[0]

def remove_urls(text):
      return re.sub(r'http\S+', 'redacted', text)
# For non-english datasets
def remove_english_words(text):
    # Looks for all English words and removes them.
    pattern = r'\b[a-zA-Z]+\b'
    return re.sub(pattern, '', text)
def delete_html_tags(text):
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text
def remove_mention(text):
  mention_regex = r"@\w+"
  return re.sub(mention_regex, "/mention", text)
def redact_email(text): 
    return re.sub(r'\S+@\S+', '/email', text)
# def remove_password(text): 
#     copy_text = text
#     pass_pattern = r'[A-Za-z0-9@#$%^&+=]{8,}'
#     text_ = re.sub(pass_pattern, '', text)
#     return text_
def remove_whitespace(text):
    return  " ".join(text.split())
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)
def sen_len_threshold(text, char_min=16, char_limit=512): # Can be used for better tuning. 
    text = str(text)
    # Removes sentences if between char_min and char_limit.
    clean_text = text if char_min <= len(text) <= char_limit else None
    return clean_text

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, ' ', data)

def filter_sensitive_words(sentence, replacement=censor_word, keys_to_filter=keys_to_filter):
    """
    Takes a list of sensitive words and replaces sensitive for you words with 'CENSORED'
    Parameters: 
        sentence 
        replacement: str = words that will be substituted instead of the sensitive words   
    """
    words = set(keys_to_filter)
    sentence_words = word_tokenize(sentence)
    
    modified_sentence = [
        replacement if word in words else word for word in sentence_words
    ]
    
    # Join the list back into a sentence
    return ' '.join(modified_sentence)

"""
In case if row has empty space " " or ""
"""
def drop_space_rows(df: pd.DataFrame, column: str ="Message") -> pd.DataFrame:
      """Identifies and drops ' ' rows in the DataFrame"""
      space_rows = (df[column] == ' ')| (df[column] == '')
      df_filtered = df[~pd.Series(space_rows)].reset_index(drop=True)

      return df_filtered

def preprocess_data(text):
      text = remove_english_words(text)
      text = redact_email(text)
      text = remove_urls(text)
      text = remove_mention(text)
      text = delete_html_tags(text)
      text = filter_sensitive_words(text)
      text = remove_whitespace(text)
      
      return text

""" Wrapper over preprocess_data function."""
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    import time 
    dataset_copy = df.copy()
    start_time= time.time()
    df['Message'] = df['Message'].apply(preprocess_data)
    df["Message"] = df["Message"].apply(lambda x: remove_emojis(str(x)) if isinstance(x, str) else ' ')
    df = drop_space_rows(df)
    print(df.head(10))
    df.to_csv(concatenated_path, index=False)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total time for processing: {total_time:.2f} seconds")

    
    return df

""" Quick function to fix the structure of the dataset for next processing function """
def structure_dataset(df: pd.DataFrame) -> pd.DataFrame:
      """ 
      Checks the dataset for mistakes and corrects them for separate_sentences function. 

      Args: 
            df: pd.DataFrame.
            Ideal dataset will contain odd rows sent by someone else, and even rows as answers by you.

      Returns:
            df: pd.DataFrame
            Dataset will contain odd rows sent by someone else, and even rows as answers by you.
      """

      dataframe: pd.DataFrame = df.copy()
      last_sent: bool = False
      previous_sender: str = "" 
      total_sins: int = 0

      def fix_row(df, idx, total_sins): 
            # Drop the row and count total
            df.loc[idx, "Message"] = None 
            total_sins += 1
            return df, total_sins 

      # Make the first row the first question (All questions become odds, answers->even)
      if dataframe["Sent_by_me"].iloc[0]: 
            dataframe = dataframe.drop(dataframe.index[0]).reset_index(drop=True)

      start_time = time.time()
      for idx, (sender, sent_by_me) in tqdm(enumerate(dataframe.loc[:, ["Sender", "Sent_by_me"]].values)):
            if sent_by_me:
                  # If there are two rows with same sender, concatenate the message into one message.
                  if sender == previous_sender or sent_by_me == last_sent:
                        # Concatenate both strings
                        dataframe.loc[idx, "Message"] = dataframe.loc[idx, "Message"] + " " + dataframe.loc[idx-1, "Message"]
                        # Delete concatanated row
                        dataframe, total_sins = fix_row(dataframe, idx-1, total_sins)

                        
            else:
                  # If there are two rows with same sender, concatenate the message into one message.
                  if sender == previous_sender:
                        # Concatenate both strings
                        dataframe.loc[idx, "Message"] = dataframe.loc[idx, "Message"] + " " +  dataframe.loc[idx-1, "Message"]
                        # Delete concatanated row
                        dataframe, total_sins = fix_row(dataframe, idx-1, total_sins)
                        continue
                  
                  # If there was a group chat, and two other people except me had a conversation
                  elif idx != 0 and dataframe.loc[idx-1, "Sent_by_me"] == False:
                        dataframe, total_sins = fix_row(dataframe, idx-1, total_sins)

                  
            last_sent = sent_by_me
            previous_sender = sender
      
      # Drop all None rows
      dataframe = dataframe.dropna().reset_index(drop=True)
      print(f"Total run time: {time.time() - start_time:.2f}. Total sins {total_sins}")
      return dataframe

""" Tester for 'structure_dataset' function above to work properly. Raises exception otherwise """
def check_structure(df: pd.DataFrame) -> pd.DataFrame:
      even_rows = df.iloc[::2]
      odd_rows = df.iloc[1::2]
      
      # Identify rows that do not meet the criteria
      sin_even_rows = even_rows[even_rows['Sent_by_me'] != False]
      sin_odd_rows = odd_rows[odd_rows['Sent_by_me'] != True]
      
      # Check if there are any sins
      if sin_even_rows.empty and sin_odd_rows.empty:
            print("All even rows are True, and all odd rows are False. Continuing processing allowed.")

      else:
            print("There are rows that don't meet the criteria:")
            if not sin_even_rows.empty:
                  print("Even rows that aren't True:")
                  print(sin_even_rows)
                  
            if not sin_odd_rows.empty:
                  print("Odd rows that aren't False:")
                  print(sin_odd_rows)

            raise Exception("Check_structure: There are rows that don't meet the criteria.")
                  

"""
Creating a column with time difference between messages 
To correctly assign the context later in processing.
"""
def create_time_diff_column(df: pd.DataFrame) -> pd.DataFrame:
      df = df.sort_values(by=['Date']).reset_index(drop=True)

      df['Date'] = pd.to_datetime(df['Date'], format='ISO8601')

      reference_time = df['Date'].min()
      df['time_diff_seconds'] = df['Date'] - reference_time
      # Converts into hours difference
      df['time_diff_seconds'] = df['time_diff_seconds'].apply(lambda x: int(x.total_seconds()))
      return df


def separate_sentences(df: pd.DataFrame) -> pd.DataFrame:
      """
      Takes a pandas dataframe with a messages column and returns separated rows with question / answer columns
      Args: 
            dataset: pd.DataFrame
            Dataset should contain a messages column and first row with identification who sent a message.
      """

      separated_dataset = pd.DataFrame(columns=['question', 'answer', 'timestamp', 'Sent_by_me', 'time_diff_seconds'])

      # Make the first row the first question (All questions become even, answers->odds)
      if df["Sent_by_me"].iloc[0]: 
            df = df.drop(df.index[0]).reset_index(drop=True)

      questions_df = df[df.index % 2 == 0].reset_index(drop=True)
      answers_df = df[df.index % 2 == 1].reset_index(drop=True)

      min_length = min(len(questions_df), len(answers_df))

      separated_dataset = pd.concat(
     [
        questions_df["Message"][:min_length].rename("question"),
        answers_df["Message"][:min_length].rename("answer"),
        df["Date"][:min_length].rename("timestamp"),
        df["Sent_by_me"][:min_length].rename("Sent_by_me"),
        df["time_diff_seconds"][:min_length].rename("time_diff_seconds")
     ], axis=1
)

      return separated_dataset

def add_context(df: pd.DataFrame, context_size: int = context_size) -> pd.DataFrame:
    """
    Add a column with previous context to the DataFrame.
    
    The context is based on the previous messages. If the time difference 
    between messages is more than 2 hours, it's considered the start of a 
    new conversation, and the first row of that new conversation will have 
    no context. Subsequent messages in the conversation will have context.
    """
    
    context_list = []
    last_time = None  # Track the last message time to determine time gaps
    
    for index in range(len(df)):
        if index == 0:
            # No context for the very first message
            context_list.append(None)
            last_time = df.loc[index, "time_diff_seconds"]
            continue
        
        # Calculate the time difference from the previous row
        time_diff = df.loc[index, "time_diff_seconds"] - last_time
        last_time = df.loc[index, "time_diff_seconds"]

        # If time_diff is more than 6 hours, consider it a new conversation
        if time_diff > 21600:
            context_list.append(None)  # Start of a new conversation, no context
        else:
            # Create context from the previous messages within the context size
            start_index = max(index - context_size, 0)
            context = df.loc[start_index:index - 1, ["question", "answer"]]

            # Build the context string from previous rows
            message = []
            for key, (question, answer) in enumerate(zip(context["question"], context["answer"])):
                message.append(f"Q{key + 1}: {question}. A{key + 1}: {answer} || ")

            # Append the concatenated message as the context
            context_list.append(" ".join(message))

    # Handle 1st row None (diff seconds in 0 index is 0, then 1 is None).
    context = df.loc[0, ["question", "answer"]]
    question, answer = context["question"], context["answer"]
    context_list[1] = (f"Q{1}: {question}. A{1}: {answer} || ")
    
    # Add the context as a new column
    df["context"] = context_list

    # Replace any empty or missing contexts with "Missing Context" if desired
    df["context"] = df["context"].apply(lambda x: "Time Gap" if pd.isna(x) else x)
    
    return df

"""  Augmentation functions !  """         
def remove_double_commas(text: str) -> str:
    """Removes double commas from the text."""
    return text.replace(",,", ",")

def split_sentences(text: str) -> list:
    """Splits the text into sentences by commas, handling empty strings gracefully."""
    return [sentence.strip() for sentence in text.split(',') if sentence.strip()]

def shuffle_sentence(text: str) -> str:
    """
    Removes double commas, splits the text into sentences, shuffles them,
    and joins them back into a shuffled sentence.
    """
    # Step 1: Clean and split the sentences
    clean_text = remove_double_commas(text)
    sentences = split_sentences(clean_text)

    # Step 2: Shuffle the sentences
    random.shuffle(sentences)

    # Step 3: Join shuffled sentences back into a single string
    return ", ".join(sentences)

def swap_word(sentence): 
    """Swaps two random words in the sentence"""
    words = word_tokenize(sentence)
    if len(words) < 2:
        return sentence

    idx1, idx2 = np.random.choice(len(words), size=2, replace=False)
    words[idx1], words[idx2] = words[idx2], words[idx1]

    return " ".join(words)

def filter_stopwords(sentence, stop_words=ukrainian_stop_words) -> str:
    """ Returns two lists: words with stopwords and words without stopwords"""
    words = word_tokenize(sentence)
    filtered_stopwords = [word for word in words if word.lower() not in stop_words]
    return words, filtered_stopwords

def pop_word(sentence, word_swap: bool = False):
    """Pops a random word from the sentence"""

    words, stop_words = filter_stopwords(sentence)

    if stop_words: 
        remove_index = np.random.choice(stop_words, size=1, replace=False)[0]
        words.remove(remove_index)
    else: 
        return sentence


    return " ".join(words)

# TODO: Add auto model download
aug_glove = naw.WordEmbsAug(
    model_type='glove', model_path=glove_path,
    action="substitute")

# TODO: Check everything below

class google_translate:
    """
    Performs Google Translate on a given text.

    Args:
        translate_from (str): The natural language of the text. Defaults to "uk". Contains auto language detection.
        translate_to (str): The language to translate to and back from. Defaults to "en".
    """

    def __init__(self, translate_from: str = "uk", translate_to: str = "en", replace_synonyms: bool = False):
        self.native_language = translate_from
        self.tunnel_language = translate_to 
        self.translator = Translator()

        if replace_synonyms:
            self.word2vec_model = self.install_word2vec()

    """ Back-translation """
    # Check whether the language input is correct
    @cache
    def check_language(self, text):
        try: 
            if self.native_language not in LANGUAGES:  
                self.native_language = self.translator.detect(text).lang
                print(f"Incorrect language. Translating from '{self.native_language}'")

                # If the back-translation is going on English text, the text will be translated from English to Spanish and back to English.
                if self.native_language == "en": 
                    self.tunnel_language = "es"

        except Exception as e:
            raise Exception("Check_language: " + str(e))

    @cache 
    def back_translate(self, text, replace_synonym: bool = True) -> str:
        """
        Performs back-translation on a given text.

        Args:
            text (str): The text to back-translate.
            temp_lang (str): The intermediate language for translation. Defaults to French ("fr").

        Returns:
            str: The back-translated text.
        """
        translator = self.translator
        try: 
            self.check_language(text=text)

            translated = self.translator.translate(text, src=self.native_language, dest=self.tunnel_language).text
            
            if replace_synonym: 
                translated = self.synonym_replacement(sentence=translated) 

            back_translated = translator.translate(translated, src=self.tunnel_language, dest=self.native_language).text

            return back_translated
        except Exception as e: 
            print("back_translate: Something went wrong.")

    """ Synonym extension (Word2Vec) """

    

    def install_word2vec(self):
      model_name = "word2vec-google-news-300"
      print(f"Configuring {model_name}")
      word2vec_model = api.load(model_name)

      return word2vec_model

    @cache
    def synonym_replacement(self, sentence, percentage: float = synonym_percentage): 
        """ Replaces random non-stopword word with a synonym. 

        Args:
            percentage (float, optional): Percentage of words to replace. Defaults to 0.7.
        
        """
        # Remove stopwords 
        words, filtered_sentence = filter_stopwords(sentence)
        if words: 
            try: 
                random_word_index = np.random.choice(len(filtered_sentence), size=int(percentage * len(filtered_sentence) if len(filtered_sentence) > 1 else 1))[0]
                word_to_replace = filtered_sentence[random_word_index]
                synonym = self.word2vec_model.most_similar(word_to_replace, topn=1)[0][0] # Top 5 most similar words
                # Fill the chosen word for a synomym
                for idx, word in enumerate(words): 
                    if word == word_to_replace: 
                        words[idx] = word_to_replace

                return " ".join(words)
            except Exception as e: 
                print(f"synonym_replacement Exception: Could not replace synonym: {str(e)}")
                return sentence        

def is_memory(threshold_gb: float = memory_threshold, delay: int = delay): 
    """
    Pauses execution when available memory is less than threshold.
    Args:f
    - threshold_gb (float): Max memory allowed in GB.
    - delay (int): Seconds to wait before rechecking memory.
    """
    available_ram = psutil.virtual_memory().available / (1024**3)
    if available_ram <= threshold_gb:
        print(available_ram)
        #print("Memory limit reached. Waiting for resources to free up...")
        time.sleep(delay)

translator = google_translate(translate_from=dataset_language, translate_to=back_translation_language, replace_synonyms=bool_synonym)
augmentation_functions = [translator.back_translate, shuffle_sentence, pop_word, swap_word]
def select_random_functions(functions=augmentation_functions, p=probs):  # Lowered probabilities for back-translation because of low-resources
    """ Returns random functions in order to apply during processing"""

    indexes = sorted(np.random.choice(len(functions), size=random.randint(1, len(functions)), replace=False, p=p))
    return [functions[index] for index in indexes]            

def apply_augmentation(sentence, random_augmentation: bool = random_augmentation) -> pd.DataFrame:
    try: 
        # Check for available memory 
        is_memory()

        if random_augmentation:
            functions = select_random_functions()
            for function in functions:
                sentence = function(sentence)
            return sentence 
        
        sentence = translator.back_translate(sentence, replace_synonym=True)
        sentence = shuffle_sentence(sentence)
        sentence = swap_word(sentence)
        sentence = pop_word(sentence)
    except Exception as e: 
        print("apply_augmentation EXCEPTION: " + str(e))
        return sentence

def speed_test(df, samples: int = 100) -> None:
    start_time = time.time()
    df["question"] = df["question"][:samples].apply(lambda x: apply_augmentation(x))
    print("--- %s seconds ---" % (time.time() - start_time))
    return 

def augment_data(df: pd.DataFrame, 
                save_path: str = None,
                augmentation_factor: int = 2, 
                random_augmentation: bool = True, 
                swap_memory: bool = True,
                samples: int = None) -> pd.DataFrame:

    """
    Augments the data by adding augmented questions.
    
    Parameters:
        df: pd.DataFrame with "question" column
        augmentation_factor: int = 5; how many times to augment each question.
        samples: int = None; How much rows to process. 
        checkpoints: bool = True; Saves the augmentation process every iteration (augmentation_factor==1Iter)
    """
    original_dataframe = df[:samples]
 

    df_augmented = original_dataframe.copy()
    df_augmented = drop_space_rows(df_augmented, column="question")
    
    for i in tqdm(range(augmentation_factor)):
        loop_dataset = original_dataframe.copy()
        loop_dataset["question"] = loop_dataset["question"].apply(lambda x: apply_augmentation(x, random_augmentation=random_augmentation))
    
        if swap_memory and i >= 1:
            df_augmented = pd.read_csv(save_path)
            

        df_augmented = pd.concat([df_augmented, loop_dataset], axis=0).reset_index(drop=True)

        if not save_path:
            save_path = "Datasets/dataset"
            df_augmented.to_csv(save_path, index=False)
            print(f"Saved into {save_path}")
            continue
            
        # For the parallel augmentation
        # Path would be chunks/chunk_1 , chunks/chunk_2 etc..
        df_augmented.to_csv(save_path, index=False)
        print(f"Saved into {save_path}")

        if swap_memory: 
            del df_augmented
    
    # Sort the dataset for sequential data.
    df_augmented = df_augmented.sort_values(by='timestamp').reset_index(drop=True)
    df_augmented.drop_duplicates(inplace=True)
    print("Augmentation completed.")


# Parallel processing
def split_dataframe(df, chunk_size):
    chunks = np.array_split(df, chunk_size)
    return chunks

def augmentation_wrapper(df: pd.DataFrame, save_path: str, worker_id: int = None, **kwargs):
      if worker_id: 
         time.sleep(worker_id * init_time) 
         
      return augment_data(df, save_path, **kwargs)

def parallel_computing(df, func, num_partitions=num_workers, num_chunks: int = num_workers, sequential_initialization=True, **kwargs):
    df_split = np.array_split(df, num_chunks) 
    save_paths = [f"data_chunks/chunk_{i+1}" for i in range(num_chunks)] # Create save_paths for each partition
    
    func_with_kwargs = partial(func, **kwargs)

    # Create a pool of workers\
    pool = None
    try:
      # Apply the function to each partition in parallel
      pool = mp.Pool(processes=num_partitions, maxtasksperchild=4) 

      if sequential_initialization:
        pool.starmap(func_with_kwargs, [(df_split[i], save_paths[i], i) for i in range(num_chunks)])
      else: 
        pool.starmap(func_with_kwargs, [(df_split[i], save_paths[i]) for i in range(num_chunks)])

    except Exception as e:
      print("parallel_computing EXCEPTION: " + str(e))
    finally:
      if pool is not None:
          pool.terminate()  # Safely terminate the pool
          pool.join()       # Wait for the worker processes to exit

def connect_chunks(chunks_folder):
    chunks = []
    for filename in os.listdir(chunks_folder):
      chunk = pd.read_csv(os.path.join(chunks_folder, filename))
      chunks.append(chunk)
    return pd.concat(chunks, axis=0) 

def split_data(df, train_size = 0.9):
    train_size = int(len(df) * train_size)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    return train_data, test_data


def main(df: pd.DataFrame = None , df_path: str = None, train_size: float = 0.9) -> pd.DataFrame:
    if not [df, df_path]:
        raise Exception("No input data provided.")

    if df_path: 
        df = pd.read_csv(concatenated_path)
    
    df = pd.DataFrame(df)
    df = preprocess_dataset(df)
    df = create_time_diff_column(df)
    df = structure_dataset(df)
    check_structure(df)
    df = separate_sentences(df)
    df = add_context(df)

    parallel_computing(df, augmentation_wrapper, num_chunks=num_chunks, sequential_initialization=True,**kwargs)

    # Finally.. save our final results
    df = connect_chunks(chunks_folder=chunks_path)

    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df.drop_duplicates(subset=['question'], inplace=True)
    df.drop(["Sent_by_me", "time_diff_seconds"], axis=1, inplace=True)

    if train_size:
        train_data, test_data = split_data(df, train_size)
        train_data.to_csv("Datasets/train_data.csv", index=False)
        test_data.to_csv("Datasets/test_data.csv", index=False)
    else:
        df.to_csv("Datasets/final_result.csv", index=False)

    return df
    

if __name__ == "__main__":
    main()