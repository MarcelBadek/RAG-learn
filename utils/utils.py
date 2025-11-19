from pathlib import Path


def load_files(path, extension):
    documents_dir = Path(path)
    return list(documents_dir.glob(f"*.{extension}"))