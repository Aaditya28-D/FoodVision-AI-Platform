from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfigCleaned:
    data_dir: Path = Path("../data/food-101/images")
    train_split_file: Path = Path("../data/food-101/meta/train_split_cleaned.txt")
    val_split_file: Path = Path("../data/food-101/meta/val_split_cleaned.txt")
    model_dir: Path = Path("models")
    weights_dir: Path = Path("models/weights")

    run_name: str = "efficientnet_b0_final"
    base_model_name: str = "efficientnet_b0"

    num_classes: int = 101
    image_size: int = 224

    batch_size: int = 32
    num_workers: int = 0

    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    device: str = "auto"
    use_pretrained: bool = True

    label_smoothing: float = 0.1
    min_learning_rate: float = 1e-6

    save_every_epoch: bool = True
    early_stopping_patience: int = 3
