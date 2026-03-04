import os
import math
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from image_model.config import Config
from image_model.transforms import get_transforms, imagenet_mean, imagenet_std
from image_model.dataset import get_datasets
from image_model.model import build_model


def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    img_tensor: (3,H,W) normalized with ImageNet mean/std
    returns: (3,H,W) in [0,1]
    """
    # Undo ImageNet normalization so images look natural when plotted
    mean = torch.tensor(imagenet_mean).view(3, 1, 1)
    std = torch.tensor(imagenet_std).view(3, 1, 1)
    x = img_tensor * std + mean
    return x.clamp(0, 1)


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if available

    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)  # ensure output directory exists

    # Load best checkpoint saved during training
    ckpt_path = os.path.join(out_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build test dataset with evaluation transforms (no augmentation)
    transforms_dict = get_transforms(img_size=cfg.img_size)
    _, _, test_ds = get_datasets(
        cfg.data_root, cfg.train_dir, cfg.val_dir, cfg.test_dir, transforms_dict
    )

    # DataLoader for iterating test images in batches
    loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,                  # keep deterministic ordering
        num_workers=cfg.num_workers,
        pin_memory=True,                # faster host->GPU copies if using CUDA
    )

    # Rebuild model architecture and load weights
    model = build_model(num_classes=len(test_ds.classes)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()  # disable dropout/batchnorm updates

    # Grid settings for the wrong-prediction visualization
    N = 24                 # total wrong examples to show
    ncols = 6              # tiles per row
    nrows = math.ceil(N / ncols)

    wrong_imgs = []
    wrong_titles = []

    # Collect up to N misclassified examples from the test set
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            preds = torch.argmax(model(imgs), dim=1)
            wrong_mask = preds != labels

            if wrong_mask.any():
                idxs = torch.where(wrong_mask)[0].tolist()
                for i in idxs:
                    t = labels[i].item()
                    p = preds[i].item()
                    wrong_imgs.append(imgs[i].detach().cpu())  # store CPU tensor for plotting
                    wrong_titles.append(f"T: {test_ds.classes[t]} | P: {test_ds.classes[p]}")
                    if len(wrong_imgs) >= N:
                        break

            if len(wrong_imgs) >= N:
                break

    if len(wrong_imgs) == 0:
        print("No wrong predictions found (unexpected).")
        return

    # Create a subplot grid and save a single image showing misclassified examples
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    if nrows == 1:
        axes = [axes]  # normalize shape

    k = 0
    for r in range(nrows):
        row_axes = axes[r] if nrows > 1 else axes[0]
        for c in range(ncols):
            ax = row_axes[c] if hasattr(row_axes, "__len__") else row_axes

            ax.axis("off")
            if k < len(wrong_imgs):
                img = denormalize(wrong_imgs[k])              # (3,H,W) in [0,1]
                img = img.permute(1, 2, 0).numpy()            # (H,W,3) for matplotlib
                ax.imshow(img)
                ax.set_title(wrong_titles[k], fontsize=10, pad=6)
            k += 1

    plt.tight_layout()
    out_path = os.path.join(out_dir, "wrong_predictions.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved wrong predictions grid to: {out_path}")


if __name__ == "__main__":
    main()