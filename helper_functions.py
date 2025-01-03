import os

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


