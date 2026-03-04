import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from image_model.config import Config
from image_model.transforms import get_transforms
from image_model.dataset import get_datasets
from image_model.model import build_model
from image_model.utils import run_one_epoch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cfg = Config()
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Saving to:", cfg.out_dir)
    os.makedirs(cfg.out_dir, exist_ok=True)

    transforms_dict = get_transforms(img_size=cfg.img_size)
    train_ds, val_ds, _ = get_datasets(
        cfg.data_root, cfg.train_dir, cfg.val_dir, cfg.test_dir, transforms_dict
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = build_model(num_classes=4).to(device)

    # Two learning rates: head vs backbone (same idea as your original script)
    head_params = list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr_backbone},
            {"params": head_params, "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )

    best_val_acc = 0.0
    best_path = os.path.join(cfg.out_dir, "best_model.pth")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, device, optimizer=None)

        print(f"Epoch {epoch}/{cfg.epochs}")
        print(f"  Train: loss={train_loss:.4f} acc={train_acc:.4f}")
        print(f"  Val  : loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"Saved new best model (val_acc={best_val_acc:.4f})")

    print(f"Done. Best val acc: {best_val_acc:.4f}")
    print(f"Best model saved to: {best_path}")


if __name__ == "__main__":
    main()