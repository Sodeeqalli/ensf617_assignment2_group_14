from dataclasses import dataclass

# Configuration class containing all hyperparameters and paths for the text model
@dataclass
class Config:
    data_root: str = "/work/TALC/ensf617_2026w/garbage_data"
    train_dir: str = "CVPR_2024_dataset_Train"
    val_dir: str   = "CVPR_2024_dataset_Val"
    test_dir: str  = "CVPR_2024_dataset_Test"

    out_dir: str = "text_output_model"  # directory to store logs, evaluation results, and checkpoints

    model_name: str = "distilbert-base-uncased"  # pretrained transformer backbone
    num_classes: int = 4  # number of classification categories

    max_len: int = 64  # maximum tokenized sequence length
    batch_size_train: int = 32
    batch_size_eval: int = 64
    num_workers: int = 2

    lr: float = 2e-5  # learning rate for transformer fine-tuning
    epochs: int = 10          # maximum training epochs (may stop earlier due to early stopping)
    patience: int = 2        # early stopping patience based on validation accuracy
    seed: int = 42           # random seed for reproducibility

    use_fast_tokenizer: bool = False  # whether to use HuggingFace fast tokenizer