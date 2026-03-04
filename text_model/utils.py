import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Set random seeds across libraries to ensure reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Ensure deterministic behavior when using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Return the best available computation device (GPU if available, otherwise CPU).
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str):
    """
    Create the directory if it does not already exist.
    """
    os.makedirs(path, exist_ok=True)