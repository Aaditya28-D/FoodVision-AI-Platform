import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ml.training.config import TrainingConfig
from ml.training.dataset import Food101DatasetFromSplit
from ml.training.history import save_training_history
from ml.training.model_factory import build_model
from ml.training.trainer import Trainer
from ml.training.transforms import get_eval_transforms, get_train_transforms


def resolve_device(device_config: str) -> str:
    if device_config == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device_config


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

    device = resolve_device(config.device)

    model = build_model(
        model_name=config.model_name,
        num_classes=config.num_classes,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_dir=config.checkpoint_dir,
    )

    history = {
        "model_name": config.model_name,
        "device": device,
        "epochs": [],
    }

    print("Starting training")
    print(f"Model: {config.model_name}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("-" * 60)

    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_acc = trainer.train_one_epoch(train_loader)
        val_loss, val_acc = trainer.validate(val_loader)

        scheduler.step(val_acc)

        trainer.save_checkpoint(
            model_name=config.model_name,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "learning_rate": current_lr,
        }
        history["epochs"].append(epoch_record)

        print(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

    save_training_history(history, config.history_path)
    print("-" * 60)
    print(f"Training history saved to: {config.history_path}")


if __name__ == "__main__":
    main()