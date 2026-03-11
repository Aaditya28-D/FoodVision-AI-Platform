from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings


@dataclass
class TrainingConfig:
    dataset_root: Path = settings.DATA_DIR / "food-101"
    train_split_file: Path = settings.DATA_DIR / "metadata" / "train.txt"
    test_split_file: Path = settings.DATA_DIR / "metadata" / "test.txt"
    class_names_file: Path = settings.DATA_DIR / "metadata" / "classes.txt"

    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    num_classes: int = 101

    model_name: str = "mobilenet_v3_large"
    use_pretrained: bool = True

    learning_rate: float = 3e-4
    num_epochs: int = 10
    weight_decay: float = 1e-4

    checkpoint_dir: Path = settings.BACKEND_DIR / "models"
    history_path: Path = settings.BACKEND_DIR / "models" / "mobilenet_v3_large_pretrained_history.json"

    device: str = "auto"
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 0.001
    resume_from_checkpoint: bool = False