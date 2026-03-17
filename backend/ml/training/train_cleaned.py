from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ml.training.config_cleaned import TrainingConfigCleaned
from ml.training.dataset import Food101Dataset
from ml.training.early_stopping import EarlyStopping
from ml.training.history import save_history
from ml.training.model_factory import build_model
from ml.training.trainer import evaluate_one_epoch, train_one_epoch
from ml.training.transforms import get_train_transforms, get_val_transforms


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main() -> None:
    config = TrainingConfigCleaned()
    device = resolve_device(config.device)

    train_dataset = Food101Dataset(
        data_dir=config.data_dir,
        split_file=config.train_split_file,
        transform=get_train_transforms(config.image_size),
    )

    val_dataset = Food101Dataset(
        data_dir=config.data_dir,
        split_file=config.val_split_file,
        transform=get_val_transforms(config.image_size),
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

    model = build_model(
        model_name=config.base_model_name,
        num_classes=config.num_classes,
        use_pretrained=config.use_pretrained,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.min_learning_rate,
    )
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        mode="max",
    )

    history: list[dict] = []
    best_val_top1 = 0.0

    print("Starting cleaned/final training")
    print(f"Run name: {config.run_name}")
    print(f"Base model: {config.base_model_name}")
    print(f"Use pretrained: {config.use_pretrained}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("-" * 60)

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_top1": round(train_metrics["top1"], 6),
            "train_top5": round(train_metrics["top5"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_top1": round(val_metrics["top1"], 6),
            "val_top5": round(val_metrics["top5"], 6),
            "lr": round(current_lr, 8),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_top1={train_metrics['top1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_top1={val_metrics['top1']:.4f} | "
            f"val_top5={val_metrics['top5']:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if config.save_every_epoch:
            epoch_path = config.model_dir / f"{config.run_name}_epoch_{epoch}.pth"
            save_checkpoint(model, epoch_path)

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]

            run_best_path = config.model_dir / f"{config.run_name}_best.pth"
            save_checkpoint(model, run_best_path)

            inference_best_path = config.weights_dir / f"{config.base_model_name}_best.pth"
            save_checkpoint(model, inference_best_path)

        scheduler.step()

        if early_stopping.step(val_metrics["top1"]):
            print("-" * 60)
            print(f"Early stopping triggered at epoch {epoch}")
            break

    history_path = config.model_dir / f"{config.run_name}_history.json"
    save_history(history, history_path)

    print("-" * 60)
    print(f"Best validation top1: {best_val_top1:.4f}")
    print(f"Training history saved to: {history_path}")
    print(f"Best inference weights saved to: {config.weights_dir / f'{config.base_model_name}_best.pth'}")


if __name__ == "__main__":
    main()
