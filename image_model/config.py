
from dataclasses import dataclass


# Configuration class used throughout the image model pipeline.
@dataclass
class Config:

    # Root directory where the dataset is stored on the TALC cluster
    data_root: str = "/work/TALC/ensf617_2026w/garbage_data"

    # Subdirectories for each dataset split
    train_dir: str = "CVPR_2024_dataset_Train"
    val_dir: str = "CVPR_2024_dataset_Val"
    test_dir: str = "CVPR_2024_dataset_Test"

    # Directory where model outputs will be saved
    out_dir: str = "image_output_model"

    # Random seed for reproducibility across runs
    seed: int = 42

    # Input image resolution used by the CNN backbone
    img_size: int = 224

    # Number of samples processed per training batch
    batch_size: int = 32

    # Number of worker processes for loading images
    num_workers: int = 2

    # Total number of training epochs
    epochs: int = 20

    # Learning rate used for the classifier head
    lr_head: float = 1e-3

    # Learning rate for the pretrained backbone
    lr_backbone: float = 5e-5

    # Weight decay for regularization (prevents overfitting)
    weight_decay: float = 1e-4

    # CNN backbone architecture used for feature extraction
    backbone: str = "mobilenet_v2"