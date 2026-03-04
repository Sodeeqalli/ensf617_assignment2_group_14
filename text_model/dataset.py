import os
import re
import torch
from torch.utils.data import Dataset

# Mapping between class names and numeric labels
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
    Optional exclude_texts removes duplicated samples across splits
    to prevent train/val/test leakage.
    """
    def __init__(self, root_dir, tokenizer, max_len=32, exclude_texts=None):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.exclude_texts = set(exclude_texts) if exclude_texts else set()

        # Iterate through dataset directory structure
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path) or class_name not in CLASS_TO_IDX:
                continue

            for filename in sorted(os.listdir(class_path)):
                if not filename.lower().endswith(".png"):
                    continue

                raw = filename[:-4]  # remove .png extension
                text = normalize_text(raw)

                # Skip samples that appear in other dataset splits
                if text in self.exclude_texts:
                    continue

                self.samples.append((text, CLASS_TO_IDX[class_name]))

        # Store class names in index order
        self.classes = [IDX_TO_CLASS[i] for i in range(len(CLASS_TO_IDX))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]

        # Tokenize text for transformer input
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
            "text": text,  # kept for debugging and evaluation analysis
        }