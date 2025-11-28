import json
from pathlib import Path


def load_files(path, extension):
    documents_dir = Path(path)
    return list(documents_dir.glob(f"*.{extension}"))

def load_test_set(test_file_path):
    with open(test_file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return raw