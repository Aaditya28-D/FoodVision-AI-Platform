from ml.inference.model_registry import ModelName

DEFAULT_STRATEGY = "ensemble"

STRATEGY_CONFIG = {
    "ensemble": {
        "label": "Ensemble (EfficientNet + ResNet)",
        "description": "Averages EfficientNet-B0 and ResNet50 predictions. Best default overall choice.",
        "type": "ensemble",
        "models": [
            ModelName.EFFICIENTNET_B0,
            ModelName.RESNET50,
        ],
    },
    "smart": {
        "label": "Smart Router",
        "description": "Uses EfficientNet-B0, ResNet50, and MobileNetV3-Large with routing logic.",
        "type": "router",
        "models": [
            ModelName.EFFICIENTNET_B0,
            ModelName.RESNET50,
            ModelName.MOBILENET_V3_LARGE,
        ],
    },
    "efficientnet_b0": {
        "label": "EfficientNet-B0",
        "description": "Single-model prediction using EfficientNet-B0.",
        "type": "single",
        "models": [ModelName.EFFICIENTNET_B0],
    },
    "resnet50": {
        "label": "ResNet50",
        "description": "Single-model prediction using ResNet50.",
        "type": "single",
        "models": [ModelName.RESNET50],
    },
    "mobilenet_v3_large": {
        "label": "MobileNetV3-Large",
        "description": "Single-model prediction using MobileNetV3-Large.",
        "type": "single",
        "models": [ModelName.MOBILENET_V3_LARGE],
    },
}


def get_default_strategy() -> str:
    return DEFAULT_STRATEGY


def get_strategy_keys() -> list[str]:
    return list(STRATEGY_CONFIG.keys())


def get_strategy_config(strategy: str) -> dict:
    if strategy not in STRATEGY_CONFIG:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return STRATEGY_CONFIG[strategy]


def get_strategy_models(strategy: str) -> list[ModelName]:
    return get_strategy_config(strategy)["models"]


def get_strategy_metadata() -> list[dict]:
    return [
        {
            "key": key,
            "label": value["label"],
            "description": value["description"],
        }
        for key, value in STRATEGY_CONFIG.items()
    ]
