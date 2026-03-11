from torchvision import models
from torch import nn


def build_efficientnet_b0(
    num_classes: int,
    use_pretrained: bool = False,
) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model