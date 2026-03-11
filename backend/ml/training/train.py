import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ml.training.config import TrainingConfig
from ml.training.dataset import Food101DatasetFromSplit
from ml.training.early_stopping import EarlyStopping
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
        use_pretrained=config.use_pretrained,
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

    early_stopper = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )

    start_epoch = 1
    if config.resume_from_checkpoint:
        resumed_epoch = trainer.load_best_checkpoint(config.model_name)
        if resumed_epoch > 0:
            start_epoch = resumed_epoch + 1
            print(f"Resumed from best checkpoint at epoch {resumed_epoch}")

    history = {
        "model_name": config.model_name,
        "use_pretrained": config.use_pretrained,
        "device": device,
        "epochs": [],
    }

    print("Starting training")
    print(f"Model: {config.model_name}")
    print(f"Use pretrained: {config.use_pretrained}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("-" * 60)

    for epoch in range(start_epoch, config.num_epochs + 1):
        train_loss, train_top1 = trainer.train_one_epoch(train_loader)
        val_loss, val_top1, val_top5 = trainer.validate(val_loader)

        scheduler.step(val_top1)

        trainer.save_checkpoint(
            model_name=config.model_name,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_top1,
            val_loss=val_loss,
            val_accuracy=val_top1,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_top1_accuracy": round(train_top1, 6),
            "val_loss": round(val_loss, 6),
            "val_top1_accuracy": round(val_top1, 6),
            "val_top5_accuracy": round(val_top5, 6),
            "learning_rate": current_lr,
        }
        history["epochs"].append(epoch_record)

        print(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_top1={train_top1:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_top1={val_top1:.4f} | "
            f"val_top5={val_top5:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if early_stopper.step(val_top1):
            print("-" * 60)
            print(f"Early stopping triggered at epoch {epoch}")
            break

    save_training_history(history, config.history_path)
    print("-" * 60)
    print(f"Training history saved to: {config.history_path}")


if __name__ == "__main__":
    main()