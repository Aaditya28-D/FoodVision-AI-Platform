from torchvision import models
from torch import nn


def build_resnet50(
    num_classes: int,
    use_pretrained: bool = False,
) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model