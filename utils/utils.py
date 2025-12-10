import datetime
import json
from pathlib import Path


def load_files(path, extension):
    documents_dir = Path(path)
    return list(documents_dir.glob(f"*.{extension}"))

def load_test_set(test_file_path):
    with open(test_file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return raw

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_current_datetime():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")