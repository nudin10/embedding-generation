import os

def count_lines_in_file(filepath: str) -> int:
    """
    Counts the number of lines in a file efficiently.

    Args:
        filepath (str): The path to the file.

    Returns:
        int: The total number of lines in the file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Counting lines in {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        line_count = sum(1 for line in f)
    return line_count
