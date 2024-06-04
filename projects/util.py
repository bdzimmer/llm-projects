"""
Basic utilities.
"""

from typing import Any, List

import json
import pickle


def read_lines(file_path: str) -> List[str]:
    """Read a file into a list of lines."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines


def save_json(obj: Any, file_path: str) -> None:
    """
    Save a data structure to a json file with nice
    default formatting.
    """
    with open(file_path, 'w') as f:
        json.dump(
            obj=obj,
            fp=f,
            indent=2,
            sort_keys=False
        )


def load_pkl(file_path: str) -> Any:
    """Load a pickle file."""
    with open(file_path, 'rb') as f:
        res = pickle.load(f)
    return res


def save_pkl(obj: Any, file_path: str) -> None:
    """Save a single object to a pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
