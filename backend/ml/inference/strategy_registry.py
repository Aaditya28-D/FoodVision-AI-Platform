from dataclasses import dataclass
from typing import Final

from ml.inference.model_registry import ModelName


@dataclass(frozen=True)
class PredictionStrategyDefinition:
    key: str
    label: str
    description: str
    category: str
    is_default: bool = False


DEFAULT_STRATEGY: Final[str] = "ensemble"

PREDICTION_STRATEGIES: Final[list[PredictionStrategyDefinition]] = [
    PredictionStrategyDefinition(
        key="ensemble",
        label="Ensemble",
        description="Averages EfficientNet-B0 and ResNet50 prediction probabilities.",
        category="ensemble",
        is_default=True,
    ),
    PredictionStrategyDefinition(
        key="smart",
        label="Smart Router",
        description="Uses router logic across EfficientNet-B0, ResNet50, and MobileNetV3-Large.",
        category="router",
    ),
    PredictionStrategyDefinition(
        key=ModelName.EFFICIENTNET_B0.value,
        label="EfficientNet-B0",
        description="Runs only the EfficientNet-B0 model.",
        category="single_model",
    ),
    PredictionStrategyDefinition(
        key=ModelName.RESNET50.value,
        label="ResNet50",
        description="Runs only the ResNet50 model.",
        category="single_model",
    ),
    PredictionStrategyDefinition(
        key=ModelName.MOBILENET_V3_LARGE.value,
        label="MobileNetV3-Large",
        description="Runs only the MobileNetV3-Large model.",
        category="single_model",
    ),
    PredictionStrategyDefinition(
        key=ModelName.VIT_B_16.value,
        label="ViT-B/16",
        description="Runs only the ViT-B/16 model.",
        category="single_model",
    ),
    PredictionStrategyDefinition(
        key=ModelName.GOOGLENET.value,
        label="GoogLeNet",
        description="Runs only the GoogLeNet model.",
        category="single_model",
    ),
]


def get_default_strategy() -> str:
    return DEFAULT_STRATEGY


def get_strategy_keys() -> list[str]:
    return [strategy.key for strategy in PREDICTION_STRATEGIES]


def get_strategy_definitions() -> list[PredictionStrategyDefinition]:
    return list(PREDICTION_STRATEGIES)


def is_valid_strategy(strategy_key: str) -> bool:
    return strategy_key in get_strategy_keys()
