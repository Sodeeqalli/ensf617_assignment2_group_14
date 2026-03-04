from dataclasses import dataclass

@dataclass
class Config:
    data_root: str = "/work/TALC/ensf617_2026w/garbage_data"
    train_dir: str = "CVPR_2024_dataset_Train"
    val_dir: str = "CVPR_2024_dataset_Val"
    test_dir: str = "CVPR_2024_dataset_Test"

    out_dir: str = "image_output_model"
    seed: int = 42

    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 2

    epochs: int = 20
    lr_head: float = 1e-3
    lr_backbone: float = 5e-5
    weight_decay: float = 1e-4

    backbone: str = "mobilenet_v2"