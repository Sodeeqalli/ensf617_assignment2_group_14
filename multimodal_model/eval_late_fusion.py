# multimodal_model/eval_late_fusion.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report

from image_model.config import Config as ImgCfg
from image_model.model import build_model as build_img_model
from image_model.transforms import get_transforms as get_img_transforms

from text_model.config import Config as TxtCfg
from text_model.model import DistilBertClassifier
from text_model.dataset import normalize_text


def list_png_names(root: str) -> set[str]:
    """Collect all .png filenames (lowercased) under class subfolders."""
    s = set()
    for cls in os.listdir(root):
        p = os.path.join(root, cls)
        if not os.path.isdir(p):
            continue
        for fn in os.listdir(p):
            if fn.lower().endswith(".png"):
                s.add(fn.lower())
    return s


@torch.no_grad()
def main(alpha: float = 0.7, split: str = "test", filter_train_overlap: bool = True) -> float:
    """
    Late fusion of image + text model probabilities.

    alpha = weight for image probs.
    p_fused = alpha * p_img + (1 - alpha) * p_txt

    split: "train" | "val" | "test"
    filter_train_overlap: when evaluating on val/test, drop any sample whose filename exists in train.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_cfg = ImgCfg()
    txt_cfg = TxtCfg()

    out_dir = "multimodal_output_model"
    os.makedirs(out_dir, exist_ok=True)

    train_root = os.path.join(img_cfg.data_root, img_cfg.train_dir)
    val_root = os.path.join(img_cfg.data_root, img_cfg.val_dir)
    test_root = os.path.join(img_cfg.data_root, img_cfg.test_dir)

    # ----- pick split root -----
    if split == "train":
        root = train_root
    elif split == "val":
        root = val_root
    else:
        root = test_root

    # ----- transforms -----
    tfms = get_img_transforms(img_size=img_cfg.img_size)
    if split == "train":
        t = tfms["train"]
    elif split == "val":
        t = tfms["val"]
    else:
        t = tfms["test"]

    # ----- dataset -----
    ds = ImageFolder(root, transform=t)

    # Optional: remove any filenames that appear in train (prevents overlap leakage if dataset has duplicates)
    if filter_train_overlap and split in ("val", "test"):
        train_names = list_png_names(train_root)
        filtered = [(p, y) for (p, y) in ds.samples if os.path.basename(p).lower() not in train_names]
        ds.samples = filtered
        ds.imgs = filtered  # ImageFolder alias

    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=img_cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    # ----- load image model -----
    img_ckpt = os.path.join(img_cfg.out_dir, "best_model.pth")
    img_model = build_img_model(num_classes=len(ds.classes)).to(device)
    img_model.load_state_dict(torch.load(img_ckpt, map_location=device))
    img_model.eval()

    # ----- load text model -----
    txt_ckpt = os.path.join(txt_cfg.out_dir, "best_model.pt")
    tokenizer = AutoTokenizer.from_pretrained(txt_cfg.model_name, use_fast=txt_cfg.use_fast_tokenizer)
    txt_model = DistilBertClassifier(txt_cfg.model_name, txt_cfg.num_classes).to(device)
    txt_model.load_state_dict(torch.load(txt_ckpt, map_location=device))
    txt_model.eval()

    # Sanity print once (class order should match your text mapping assumption)
    # Expected: ['Black', 'Blue', 'Green', 'TTR']
    print("Classes:", ds.classes)
    print(f"Split={split} samples={len(ds)} alpha={alpha:.2f} filter_train_overlap={filter_train_overlap}")

    softmax = nn.Softmax(dim=1)

    y_true, y_pred = [], []

    # ImageFolder stores file paths in ds.samples
    sample_paths = [p for (p, _) in ds.samples]
    ptr = 0

    for imgs, labels in loader:
        b = imgs.size(0)
        batch_paths = sample_paths[ptr:ptr + b]
        ptr += b

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # image probs
        p_img = softmax(img_model(imgs))

        # text probs from filenames
        texts = [normalize_text(os.path.splitext(os.path.basename(p))[0]) for p in batch_paths]
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=txt_cfg.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        p_txt = softmax(txt_model(input_ids, attention_mask))

        # fuse
        p_fused = alpha * p_img + (1 - alpha) * p_txt
        preds = torch.argmax(p_fused, dim=1)

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    acc = float((y_true_np == y_pred_np).mean())
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(ds.classes))))
    report = classification_report(y_true_np, y_pred_np, target_names=ds.classes, digits=4)

    out_path = os.path.join(out_dir, f"late_fusion_{split}_alpha{alpha:.2f}.txt")
    with open(out_path, "w") as f:
        f.write(f"split={split}\nalpha={alpha}\nacc={acc:.4f}\n")
        f.write(f"filter_train_overlap={filter_train_overlap}\n")
        f.write(f"samples={len(ds)}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")

    print(f"✅ Late fusion ({split}) alpha={alpha:.2f} acc={acc:.4f}")
    print(f"Saved: {out_path}")
    return acc


if __name__ == "__main__":
    # ----- sweep alpha on validation -----
    best_alpha = None
    best_acc = -1.0

    print("Sweeping alpha on validation set...")
    for alpha in np.linspace(0.0, 1.0, 11):
        acc = main(alpha=float(alpha), split="val", filter_train_overlap=True)
        if acc > best_acc:
            best_acc = acc
            best_alpha = float(alpha)

    print(f"\nBest alpha from validation: {best_alpha:.2f} (acc={best_acc:.4f})")

    # ----- final evaluation on test -----
    print("\nRunning final evaluation on TEST with best alpha...")
    main(alpha=float(best_alpha), split="test", filter_train_overlap=True)