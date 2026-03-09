from dataclasses import dataclass
from typing import Any

from ml.inference.model_registry import ModelName


@dataclass
class LoadedModel:
    model_name: str
    model: Any
    device: str
    input_size: int


class ModelLoader:
    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}

    def load_model(self, model_name: ModelName) -> LoadedModel:
        if model_name.value in self._models:
            return self._models[model_name.value]

        loaded_model = LoadedModel(
            model_name=model_name.value,
            model=None,
            device="cpu",
            input_size=224,
        )

        self._models[model_name.value] = loaded_model
        return loaded_model

    def get_available_models(self) -> list[str]:
        return [model.value for model in ModelName]