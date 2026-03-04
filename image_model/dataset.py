import os
from torchvision.datasets import ImageFolder


def get_datasets(data_root, train_dir, val_dir, test_dir, transforms_dict):
    train_path = os.path.join(data_root, train_dir)
    val_path = os.path.join(data_root, val_dir)
    test_path = os.path.join(data_root, test_dir)

    train_ds = ImageFolder(train_path, transform=transforms_dict["train"])
    val_ds = ImageFolder(val_path, transform=transforms_dict["val"])
    test_ds = ImageFolder(test_path, transform=transforms_dict["test"])

    return train_ds, val_ds, test_ds