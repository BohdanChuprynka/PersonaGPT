import os
import re

def find_repository_folder(start_path: str = None) -> str:
    """
    Locate the root folder of the repository by traversing upwards 
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
    if pattern is None:
        pattern = r'.*'  # Default: Match all directories

    found_dirs = []
    regex = re.compile(pattern)
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            if regex.match(dir_name):
                found_dirs.append(str(os.path.join(root, dir_name)))

    return found_dirs

