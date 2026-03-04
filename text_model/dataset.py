import os
import re
import torch
from torch.utils.data import Dataset

CLASS_TO_IDX = {"Black": 0, "Blue": 1, "Green": 2, "TTR": 3}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


def normalize_text(filename_no_ext: str) -> str:
    """
    Normalize filename-derived text to reduce noisy variations.
    Keeps it lightweight (no heavy NLP).
    """
    s = filename_no_ext.strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


class GarbageTextDataset(Dataset):
    """
    Builds (text,label) pairs from class subfolders.
    Optional exclude_texts lets us remove any text that appears in other splits
    (prevents leakage when the dataset itself has duplicates across train/val/test).
    """
    def __init__(self, root_dir, tokenizer, max_len=32, exclude_texts=None):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.exclude_texts = set(exclude_texts) if exclude_texts else set()

        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path) or class_name not in CLASS_TO_IDX:
                continue

            for filename in sorted(os.listdir(class_path)):
                if not filename.lower().endswith(".png"):
                    continue

                raw = filename[:-4]  # remove .png
                text = normalize_text(raw)

                if text in self.exclude_texts:
                    continue

                self.samples.append((text, CLASS_TO_IDX[class_name]))

        self.classes = [IDX_TO_CLASS[i] for i in range(len(CLASS_TO_IDX))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "text": text,  # helpful for eval/debug
        }