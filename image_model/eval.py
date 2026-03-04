import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from image_model.config import Config
from image_model.transforms import get_transforms
from image_model.dataset import get_datasets
from image_model.model import build_model


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = os.path.join(out_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    transforms_dict = get_transforms(img_size=cfg.img_size)
    _, _, test_ds = get_datasets(
        cfg.data_root, cfg.train_dir, cfg.val_dir, cfg.test_dir, transforms_dict
    )

    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = build_model(num_classes=4).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=test_ds.classes,
        digits=4
    )
    acc = (np.array(y_true) == np.array(y_pred)).mean()

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=test_ds.classes,
        yticklabels=test_ds.classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # ----------------------------
    # SAVE SUMMARY TXT
    # ----------------------------
    summary_path = os.path.join(out_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Using device: {device}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Test accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")

    print(f"Test accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    print(f"\nSaved outputs to ./{out_dir}/")
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()