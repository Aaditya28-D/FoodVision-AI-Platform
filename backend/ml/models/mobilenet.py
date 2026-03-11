from torchvision import models
from torch import nn


def build_mobilenet_v3_large(
    num_classes: int,
    use_pretrained: bool = False,
) -> nn.Module:
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if use_pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model