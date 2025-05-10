import os

def batch_read_jsonl(file_path: str, batch_size:int=10000):
    """
    Reads a newline-delimited JSON (JSONL) file line by line and yields
    batches of lines as lists of strings.

    Args:
        file_path: The path to the JSONL file.
        batch_size: The maximum number of lines to include in each batch.
                    Defaults to 10000.

    Yields:
        A list of strings, where each string is a line from the JSONL file.
        The last batch may contain fewer lines than batch_size.
    """

    current_batch = []
    batch_counter = 0

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return 

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                current_batch.append(line.strip())
                batch_counter += 1

                if batch_counter == batch_size:
                    yield current_batch
                    current_batch = []
                    batch_counter = 0
                
            if current_batch:
                yield current_batch
    except:
        raise
