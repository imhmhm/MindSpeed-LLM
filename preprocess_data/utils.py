import glob
import os

def get_input_paths(input_path):
    if '*' in input_path or '?' in input_path:
        paths = sorted(glob.glob(input_path))
    elif os.path.isfile(input_path):
        paths = [input_path]
    elif os.path.isdir(input_path):
        paths = []
        for pattern in ("*.json*", "*.parquet", "*.parq"):
            paths.extend(glob.glob(os.path.join(input_path, pattern)))
        paths = sorted(paths)
    else:
        paths = []

    if not paths:
        raise ValueError(f"No files found in {input_path}")
    return paths