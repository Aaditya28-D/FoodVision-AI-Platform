from torchvision import models
from torch import nn


def build_googlenet(
    num_classes: int,
    use_pretrained: bool = False,
) -> nn.Module:
    weights = models.GoogLeNet_Weights.DEFAULT if use_pretrained else None
    model = models.googlenet(weights=weights, aux_logits=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model