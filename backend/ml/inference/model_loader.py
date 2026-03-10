from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.core.config import settings
from ml.inference.model_registry import ModelName
from ml.models.googlenet import build_googlenet
from ml.models.mobilenet import build_mobilenet_v3_large


@dataclass
class LoadedModel:
    model_name: str
    model: Any
    device: str
    input_size: int
    weights_path: str | None


class ModelLoader:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self._models: dict[str, LoadedModel] = {}

    def _get_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _build_model_and_path(self, model_name: ModelName) -> tuple[Any, Path]:
        if model_name == ModelName.MOBILENET_V3_LARGE:
            return (
                build_mobilenet_v3_large(self.num_classes),
                settings.MOBILENET_WEIGHTS_PATH,
            )

        if model_name == ModelName.GOOGLENET:
            return (
                build_googlenet(self.num_classes),
                settings.GOOGLENET_WEIGHTS_PATH,
            )

        raise ValueError(f"Unsupported model: {model_name}")

    def load_model(self, model_name: ModelName) -> LoadedModel:
        if model_name.value in self._models:
            return self._models[model_name.value]

        device = self._get_device()
        model, weights_path = self._build_model_and_path(model_name)

        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        loaded_model = LoadedModel(
            model_name=model_name.value,
            model=model,
            device=device,
            input_size=224,
            weights_path=str(weights_path) if weights_path.exists() else None,
        )

        self._models[model_name.value] = loaded_model
        return loaded_model

    def get_available_models(self) -> list[str]:
        return [model.value for model in ModelName]