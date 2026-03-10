from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str | Path,
    ) -> None:
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_accuracy = 0.0

        self.model.to(self.device)

    def train_one_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        return epoch_loss, epoch_accuracy

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> tuple[float, float]:
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        return epoch_loss, epoch_accuracy

    def save_checkpoint(
        self,
        model_name: str,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_accuracy: float,
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "best_val_accuracy": self.best_val_accuracy,
            "device": self.device,
        }

        checkpoint_path = self.checkpoint_dir / f"{model_name}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            checkpoint["best_val_accuracy"] = self.best_val_accuracy
            best_model_path = self.checkpoint_dir / f"{model_name}_best.pth"
            torch.save(checkpoint, best_model_path)

    def load_best_checkpoint(self, model_name: str) -> int:
        best_model_path = self.checkpoint_dir / f"{model_name}_best.pth"

        if not best_model_path.exists():
            return 0

        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)

        return int(checkpoint.get("epoch", 0))