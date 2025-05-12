import os
import shutil

def delete_huggingface_cache_directory():
    """
    WARNING: Irreversible
    Deletes the default Hugging Face disk cache directory entirely.
    This is to save disk space.
    """
    
    # Get the user's home directory
    home_dir = os.path.expanduser('~')

    cache_path = os.path.join(home_dir, '.cache', 'huggingface')

    print(f"Attempting to delete cache directory: {cache_path}")

    if os.path.exists(cache_path):
        try:
            shutil.rmtree(cache_path)
            print("Hugging Face cache directory deleted successfully.")
        except OSError as e:
            print(f"Error deleting directory {cache_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        print("Hugging Face cache directory does not exist at the default location.")
        