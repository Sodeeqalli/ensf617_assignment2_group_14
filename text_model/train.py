import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm

from text_model.config import Config
from text_model.dataset import GarbageTextDataset
from text_model.model import DistilBertClassifier
from text_model.utils import set_seed, get_device, ensure_dir


def collect_texts(root_dir: str) -> set[str]:
    """Collect filename texts (without .png) from a split root across all class folders."""
    texts = set()
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            if filename.lower().endswith(".png"):
                texts.add(filename[:-4])
    return texts


# Compute accuracy from model logits
def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


# Run one training or evaluation epoch
def run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in tqdm(loader, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = ce(logits, labels)
        acc = accuracy_from_logits(logits, labels)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)  # clear gradients
            loss.backward()                       # backpropagation
            optimizer.step()                      # update model parameters

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def main():
    cfg = Config()
    set_seed(cfg.seed)  # ensure reproducibility

    device = get_device()
    print(f"Using device: {device}")

    out_dir = cfg.out_dir
    ensure_dir(out_dir)

    ckpt_path = os.path.join(out_dir, "best_model.pt")
    train_log_path = os.path.join(out_dir, "train_log.txt")

    train_root = os.path.join(cfg.data_root, cfg.train_dir)
    val_root = os.path.join(cfg.data_root, cfg.val_dir)
    test_root = os.path.join(cfg.data_root, cfg.test_dir)

    # Initialize tokenizer matching the pretrained model
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, use_fast=cfg.use_fast_tokenizer
    )

    # ---- Prevent dataset leakage across splits ----
    val_texts = collect_texts(val_root)
    test_texts = collect_texts(test_root)
    exclude_texts = val_texts | test_texts

    # Build datasets
    train_ds = GarbageTextDataset(
        train_root, tokenizer, max_len=cfg.max_len, exclude_texts=exclude_texts
    )
    val_ds = GarbageTextDataset(val_root, tokenizer, max_len=cfg.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size_train,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size_eval,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Build DistilBERT classifier
    model = DistilBertClassifier(cfg.model_name, cfg.num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    best_val_acc = 0.0
    bad_epochs = 0

    # Write training metadata to log
    with open(train_log_path, "w") as f:
        f.write(f"Using device: {device}\n")
        f.write(f"Model: {cfg.model_name}\n")
        f.write(f"Train: {train_root}\nVal: {val_root}\nTest: {test_root}\n")
        f.write(f"Output: {out_dir}\n\n")
        f.write("Leakage filter:\n")
        f.write(f"  val_texts={len(val_texts)} test_texts={len(test_texts)}\n")
        f.write(f"  excluded_from_train={len(exclude_texts)} (union)\n")
        f.write(f"  train_samples_after_filter={len(train_ds)}\n\n")

    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None)

        msg = (
            f"Epoch {epoch}/{cfg.epochs}\n"
            f"  Train: loss={train_loss:.4f} acc={train_acc:.4f}\n"
            f"  Val  : loss={val_loss:.4f} acc={val_acc:.4f}\n"
        )
        print(msg, end="")

        with open(train_log_path, "a") as f:
            f.write(msg)

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)

            save_msg = f"  Saved new best model (val_acc={best_val_acc:.4f})\n"
            print(save_msg, end="")
            with open(train_log_path, "a") as f:
                f.write(save_msg)
        else:
            bad_epochs += 1

            # Early stopping if validation does not improve
            if bad_epochs >= cfg.patience:
                stop_msg = f"Early stopping (no improvement for {cfg.patience} epochs).\n"
                print(stop_msg, end="")
                with open(train_log_path, "a") as f:
                    f.write(stop_msg)
                break

    done_msg = f"\nDone. Best val acc: {best_val_acc:.4f}\n"
    print(done_msg, end="")
    with open(train_log_path, "a") as f:
        f.write(done_msg)


if __name__ == "__main__":
    main()