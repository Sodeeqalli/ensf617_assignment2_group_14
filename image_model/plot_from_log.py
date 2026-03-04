import re
import os
import matplotlib.pyplot as plt
from image_model.config import Config

def main():
    cfg = Config()
    log_path = os.path.join(cfg.out_dir, "train_log.txt")

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log not found: {log_path}")

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    train_pat = re.compile(r"Train:\s+loss=([0-9.]+)\s+acc=([0-9.]+)")
    val_pat   = re.compile(r"Val\s+:\s+loss=([0-9.]+)\s+acc=([0-9.]+)")

    with open(log_path) as f:
        for line in f:
            t = train_pat.search(line)
            v = val_pat.search(line)
            if t:
                train_loss.append(float(t.group(1)))
                train_acc.append(float(t.group(2)))
            if v:
                val_loss.append(float(v.group(1)))
                val_acc.append(float(v.group(2)))

    epochs = range(1, len(train_loss) + 1)

    # Loss plot
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "train_val_loss.png"), dpi=200)
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "train_val_acc.png"), dpi=200)
    plt.close()

    print("Saved train_val_loss.png")
    print("Saved train_val_acc.png")

if __name__ == "__main__":
    main()