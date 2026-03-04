import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from text_model.config import Config
from text_model.dataset import GarbageTextDataset, IDX_TO_CLASS
from text_model.model import DistilBertClassifier
from text_model.utils import get_device, ensure_dir


def save_confusion_matrix_png(cm, classes, out_path: str):
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    # write counts
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=8)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = Config()
    device = get_device()

    out_dir = cfg.out_dir
    ensure_dir(out_dir)

    ckpt_path = os.path.join(out_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    test_root = os.path.join(cfg.data_root, cfg.test_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, use_fast=cfg.use_fast_tokenizer
    )

    test_ds = GarbageTextDataset(test_root, tokenizer, max_len=cfg.max_len)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size_eval,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = DistilBertClassifier(cfg.model_name, cfg.num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    wrong_examples = {}  # true_idx -> (text, true, pred)

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            texts = batch["text"]  # list[str] from dataset

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            for t, p, txt in zip(labels.tolist(), preds.tolist(), texts):
                y_true.append(t)
                y_pred.append(p)

                if p != t and t not in wrong_examples:
                    wrong_examples[t] = (txt, t, p)

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    acc = float((y_true_np == y_pred_np).mean())

    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(cfg.num_classes)))
    report = classification_report(
        y_true_np, y_pred_np,
        target_names=[IDX_TO_CLASS[i] for i in range(cfg.num_classes)],
        digits=4
    )

    # save summary txt
    summary_path = os.path.join(out_dir, "test_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Using device: {device}\n")
        f.write(f"Model: {cfg.model_name}\n")
        f.write(f"Checkpoint: {ckpt_path}\n\n")
        f.write(f"✅ Test accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix (rows=true, cols=pred):\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Representative wrong examples (one per true class):\n")
        for cls_idx in range(cfg.num_classes):
            if cls_idx in wrong_examples:
                txt, t, p = wrong_examples[cls_idx]
                f.write(f"- True={IDX_TO_CLASS[t]} Pred={IDX_TO_CLASS[p]} Text='{txt}'\n")
            else:
                f.write(f"- True={IDX_TO_CLASS[cls_idx]}: (no misclassifications found)\n")

    # save confusion matrix png
    cm_png_path = os.path.join(out_dir, "confusion_matrix.png")
    save_confusion_matrix_png(cm, [IDX_TO_CLASS[i] for i in range(cfg.num_classes)], cm_png_path)

    print(f" Test accuracy: {acc:.4f}")
    print(f" Saved {summary_path}")
    print(f" Saved {cm_png_path}")


if __name__ == "__main__":
    main()