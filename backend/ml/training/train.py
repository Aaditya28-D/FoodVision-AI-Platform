from pathlib import Path

from torch.utils.data import DataLoader

from ml.training.config import TrainingConfig
from ml.training.dataset import Food101DatasetFromSplit
from ml.training.transforms import get_eval_transforms, get_train_transforms


def main() -> None:
    config = TrainingConfig()

    images_root = config.dataset_root / "images"

    train_dataset = Food101DatasetFromSplit(
        images_root=images_root,
        split_file=config.train_split_file,
        transform=get_train_transforms(config.image_size),
    )

    val_dataset = Food101DatasetFromSplit(
        images_root=images_root,
        split_file=config.test_split_file,
        transform=get_eval_transforms(config.image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    print("Training pipeline initialized successfully")
    print(f"Model: {config.model_name}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Checkpoint directory: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()