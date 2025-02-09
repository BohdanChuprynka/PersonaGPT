import os
import re
import pandas as pd
import subprocess
import sys

def install_requirements():
    try:
        requirements_path = find_files(base_dir=".", pattern="requirements.txt")[0]
        print(requirements_path)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("All libraries from requirements.txt have been installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install libraries: {e}")

def find_repository_folder(start_path: str = None) -> str:
    """
    Locates the root folder of the repository by traversing upwards 
    until a marker file or folder is found.

    :param start_path: The starting directory for the search. 
                       Defaults to the current working directory.
    :return: The path to the repository root.
    """
    if start_path is None:
        start_path = os.getcwd()

    current_path = os.path.abspath(start_path)

    while True:
        # Check for a .git folder or a specific marker file
        if os.path.isdir(os.path.join(current_path, ".git")):
            return current_path

        # If we reach the root directory without finding the marker, stop
        parent_path = os.path.dirname(current_path)
        if current_path == parent_path:
            raise FileNotFoundError("Repository root not found.")

        # Move up one level
        current_path = parent_path

def find_files(base_dir, pattern: str = None) -> list:
    """
    Automatically locates specific files in base_dir based on pattern.

    Parameters:
    - base_dir (str): The starting directory for the search.
    - patterns (str): Name of the file

    Returns:
    - list: A list of file paths that match the specified pattern.

    Example:
    >>> find_files(base_dir="/path/to/directory", pattern="*.txt")
    ['/path/to/directory/file1.txt', '/path/to/directory/file2.txt']
    """
    if pattern is None:
        pattern = r'.*'  # Default: Match all files

    found_files = []
    regex = re.compile(pattern)
    for root, _, files in os.walk(base_dir):
        for file in files:
            if regex.match(file):
                found_files.append(str(os.path.join(root, file)))

    return found_files
    
def find_dirs(base_dir, pattern: str = None) -> list: 
    """
    Automatically locates specific directory in base_dir based on pattern.

    Parameters:
    - base_dir (str): The starting directory for the search.
    - patterns (str): Name of the directory

    Returns:
    - list: A list of directory paths that match the specified pattern. 
    """
    if not base_dir or not os.path.exists(base_dir):
        # Go from repository root_dir
        base_dir = find_repository_folder()

    if pattern is None:
        pattern = r'.*'  # Default: Match all directories

    found_dirs = []
    regex = re.compile(pattern)
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            if regex.match(dir_name):
                found_dirs.append(str(os.path.join(root, dir_name)))

    return found_dirs

def save_dataset(content: pd.DataFrame, path: str):
    """
    Provides a convenient attention to saving a dataset into the specified path

    Parameters:
    - content (str): The content to be saved.
    - path (str): The path to save the content.

    Returns:
    - None
    """

    if not os.path.exists(path):
        print(f"Saving to {path}")
        content.to_csv(path, index=False)
    else: 
        exc = f"Dataset already exists at {path}"
        while True:
                response = input(f"{exc}. Type 'overwrite' to overwrite the file, 'q' to quit, or enter a new name of the file: ")
                if response.lower() == 'overwrite':
                    print(f"Overwriting {path}")
                    content.to_csv(path, index=False)
                    break
                elif response.lower() == 'q':
                    break
                elif response.lower() == '':
                    exc = "Provided input is invalid"
                else:
                    if not response.endswith('.csv'): # Make sure that saved file is .csv
                            response += '.csv'

                    # Go down the directory to get the directory where the dataset should be saved
                    dir_path = os.path.dirname(path)
                    save_path = os.path.join(dir_path, str(response))
                    if os.path.exists(save_path):
                            exc = f"File already exists at {save_path}" # In case if file on new path already exists
                            continue
                    else:
                            print(f"Saving to {save_path}")
                            content.to_csv(save_path, index=False)
                            break

def change_prompts(language: str = "en", df: pd.DataFrame = None):
    """
    Changes the prompts in context for model requirements based on the language of the dataset.
    language: str: The native language of the dataset, will return the prompts on the specified language. USE language codes!
    df: pd.DataFrame: If specified, will change the column "context" in the dataframe to the specified language.
    """
    if language.lower() == "uk":  # type: ignore
        q_prompt = "Питання"
        c_prompt = "Контекст"
        finetune_prompt = "Підказка"
        context_label = "Відсутній контекст"
    elif language.lower() == "en":
        q_prompt = "Question"
        finetune_prompt = "Prompt"
        c_prompt = "Context"

    if df is not None:
        if not df.empty:
            df["context"] = [context_label if x == "Time Gap" else x for x in df["context"]]
            return df, q_prompt, finetune_prompt, c_prompt

    return q_prompt, finetune_prompt, c_prompt, context_label

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_repeated_chars(text):
        return re.sub(r"\s+", " ", text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def strip(text):
        return text.strip()
    
    def lower(text):
        return text.lower()

    return remove_repeated_chars(white_space_fix(strip(lower(s))))