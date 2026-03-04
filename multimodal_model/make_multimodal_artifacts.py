# multimodal_model/make_multimodal_artifacts.py
import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report

from image_model.config import Config as ImgCfg
from image_model.model import build_model as build_img_model
from image_model.transforms import get_transforms as get_img_transforms

from text_model.config import Config as TxtCfg
from text_model.model import DistilBertClassifier
from text_model.dataset import normalize_text


def list_png_names(root: str) -> set[str]:
    s = set()
    for cls in os.listdir(root):
        p = os.path.join(root, cls)
        if not os.path.isdir(p):
            continue
        for fn in os.listdir(p):
            if fn.lower().endswith(".png"):
                s.add(fn.lower())
    return s


def denormalize_imagenet(img_tensor: torch.Tensor) -> torch.Tensor:
    # ImageNet mean/std (same as your image_model.transforms)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = img_tensor * std + mean
    return x.clamp(0, 1)


def save_confusion_matrix_png(cm: np.ndarray, classes: list[str], out_path: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Multimodal Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8
            )

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def main(
    split: str = "test",
    alpha: float = 0.30,
    filter_train_overlap: bool = True,
    N_wrong: int = 24,
    ncols: int = 6,
):
    """
    Generates multimodal artifacts for a given split:
      - confusion_matrix_multimodal.png
      - multimodal_summary.txt (acc + cm + classification report)
      - wrong_predictions_multimodal.png (grid of misclassified examples)

    alpha: image weight in late fusion.
    p_fused = alpha * p_img + (1-alpha) * p_txt
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_cfg = ImgCfg()
    txt_cfg = TxtCfg()

    out_dir = "multimodal_output_model"
    os.makedirs(out_dir, exist_ok=True)

    train_root = os.path.join(img_cfg.data_root, img_cfg.train_dir)
    val_root = os.path.join(img_cfg.data_root, img_cfg.val_dir)
    test_root = os.path.join(img_cfg.data_root, img_cfg.test_dir)

    if split == "train":
        root = train_root
    elif split == "val":
        root = val_root
    else:
        root = test_root

    tfms = get_img_transforms(img_size=img_cfg.img_size)
    if split == "train":
        t = tfms["train"]
    elif split == "val":
        t = tfms["val"]
    else:
        t = tfms["test"]

    ds = ImageFolder(root, transform=t)

    if filter_train_overlap and split in ("val", "test"):
        train_names = list_png_names(train_root)
        filtered = [(p, y) for (p, y) in ds.samples if os.path.basename(p).lower() not in train_names]
        ds.samples = filtered
        ds.imgs = filtered

    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=img_cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Load image model
    img_ckpt = os.path.join(img_cfg.out_dir, "best_model.pth")
    img_model = build_img_model(num_classes=len(ds.classes)).to(device)
    img_model.load_state_dict(torch.load(img_ckpt, map_location=device))
    img_model.eval()

    # Load text model
    txt_ckpt = os.path.join(txt_cfg.out_dir, "best_model.pt")
    tokenizer = AutoTokenizer.from_pretrained(txt_cfg.model_name, use_fast=txt_cfg.use_fast_tokenizer)
    txt_model = DistilBertClassifier(txt_cfg.model_name, txt_cfg.num_classes).to(device)
    txt_model.load_state_dict(torch.load(txt_ckpt, map_location=device))
    txt_model.eval()

    softmax = nn.Softmax(dim=1)

    # For mapping batch -> filenames
    sample_paths = [p for (p, _) in ds.samples]
    ptr = 0

    y_true, y_pred = [], []

    wrong_imgs = []
    wrong_titles = []
    wrong_texts = []

    for imgs, labels in loader:
        b = imgs.size(0)
        batch_paths = sample_paths[ptr:ptr + b]
        ptr += b

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # image probs
        p_img = softmax(img_model(imgs))

        # text probs
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

        # collect metrics
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

        # collect wrong examples for grid
        if len(wrong_imgs) < N_wrong:
            wrong_mask = preds != labels
            if wrong_mask.any():
                idxs = torch.where(wrong_mask)[0].tolist()
                for i in idxs:
                    if len(wrong_imgs) >= N_wrong:
                        break
                    t_idx = labels[i].item()
                    p_idx = preds[i].item()
                    wrong_imgs.append(imgs[i].detach().cpu())
                    wrong_titles.append(f"T: {ds.classes[t_idx]} | P: {ds.classes[p_idx]}")
                    wrong_texts.append(texts[i])

    # ----- summary metrics -----
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    acc = float((y_true_np == y_pred_np).mean())
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(ds.classes))))
    report = classification_report(y_true_np, y_pred_np, target_names=ds.classes, digits=4)

    # Save confusion matrix PNG
    cm_path = os.path.join(out_dir, f"confusion_matrix_multimodal_{split}_alpha{alpha:.2f}.png")
    save_confusion_matrix_png(cm, ds.classes, cm_path)

    # Save summary txt
    summary_path = os.path.join(out_dir, f"multimodal_summary_{split}_alpha{alpha:.2f}.txt")
    with open(summary_path, "w") as f:
        f.write(f"split={split}\nalpha={alpha}\nacc={acc:.4f}\n")
        f.write(f"filter_train_overlap={filter_train_overlap}\n")
        f.write(f"samples={len(ds)}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")

    # Save wrong predictions grid
    if len(wrong_imgs) > 0:
        nrows = math.ceil(N_wrong / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
        if nrows == 1:
            axes = np.array([axes])  # normalize shape

        k = 0
        for r in range(nrows):
            for c in range(ncols):
                ax = axes[r, c]
                ax.axis("off")
                if k < len(wrong_imgs):
                    img = denormalize_imagenet(wrong_imgs[k])
                    img = img.permute(1, 2, 0).numpy()
                    ax.imshow(img)
                    # add both title and filename-text
                    ax.set_title(wrong_titles[k], fontsize=10, pad=6)
                    ax.text(
                        0.5, -0.12,
                        wrong_texts[k],
                        transform=ax.transAxes,
                        ha="center", va="top",
                        fontsize=8
                    )
                k += 1

        plt.tight_layout()
        wrong_path = os.path.join(out_dir, f"wrong_predictions_multimodal_{split}_alpha{alpha:.2f}.png")
        plt.savefig(wrong_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        wrong_path = None

    print(f"✅ Multimodal artifacts saved in: {out_dir}")
    print(f"   - acc={acc:.4f}  split={split}  alpha={alpha:.2f}  samples={len(ds)}")
    print(f"   - {cm_path}")
    print(f"   - {summary_path}")
    if wrong_path:
        print(f"   - {wrong_path}")
    else:
        print("   - No wrong predictions found (unexpected).")


if __name__ == "__main__":
    # match your best alpha from validation
    main(split="test", alpha=0.30, filter_train_overlap=True, N_wrong=24, ncols=6)