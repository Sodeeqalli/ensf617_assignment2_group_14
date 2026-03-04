from dataclasses import dataclass

@dataclass
class Config:
    data_root: str = "/work/TALC/ensf617_2026w/garbage_data"
    train_dir: str = "CVPR_2024_dataset_Train"
    val_dir: str   = "CVPR_2024_dataset_Val"
    test_dir: str  = "CVPR_2024_dataset_Test"

    out_dir: str = "text_output_model"

    model_name: str = "distilbert-base-uncased"
    num_classes: int = 4

    max_len: int = 64
    batch_size_train: int = 32
    batch_size_eval: int = 64
    num_workers: int = 2

    lr: float = 2e-5
    epochs: int = 10          # max epochs; early stopping may stop earlier
    patience: int = 2        # early stop if val_acc stops improving
    seed: int = 42

    use_fast_tokenizer: bool = False