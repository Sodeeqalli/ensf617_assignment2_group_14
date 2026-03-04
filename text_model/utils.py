import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Make training deterministic for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Return the best available device.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)