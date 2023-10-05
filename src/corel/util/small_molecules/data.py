"""Utilities for loading the dataset, and manipulating the small molecules data."""
from typing import Dict
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.resolve()


def load_zinc_250k_dataset() -> np.ndarray:
    """Returns the small molecule dataset of one-hot encoded SELFIES strings.

    Using the alphabet computed during preprocessing, this method
    loads the dataset of SELFIES strings, and one-hot encodes them.
    """
    dataset_path = (
        ROOT_DIR
        / "experiments"
        / "assets"
        / "data"
        / "small_molecules"
        / "processed"
        / "zinc250k_onehot_and_integers.npz"
    )

    return np.load(dataset_path)["onehot"]


def load_zinc_250k_alphabet() -> Dict[str, int]:
    """
    Returns the alphabet (dict[str, int]) of SELFIES characters.
    """
    alphabet_path = (
        ROOT_DIR
        / "experiments"
        / "assets"
        / "data"
        / "small_molecules"
        / "processed"
        / "alphabet_stoi.json"
    )

    with open(alphabet_path, "r") as f:
        alphabet = json.load(f)

    return alphabet


if __name__ == "__main__":
    onehot = load_zinc_250k_dataset()
    print(onehot.shape)
