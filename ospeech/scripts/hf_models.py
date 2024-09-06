from huggingface_hub import HfApi

def list_files_from_repo(repo_id, revision="main"):
    """
    List all files in a Hugging Face repository and their full paths.

    Parameters:
    - repo_id (str): The ID of the repository (e.g., "username/repo_name").
    - revision (str): The branch or tag name (default is "main").

    Returns:
    - List of tuples where each tuple contains the file name and its full path.
    """
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, revision=revision)
    return files
    
if __name__ == "__main__":
    repo_id = 'mush42/optispeech'
    print(list_files_from_repo(repo_id))
