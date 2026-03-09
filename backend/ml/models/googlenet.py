from torchvision import models
from torch import nn


def build_googlenet(num_classes: int) -> nn.Module:
    model = models.googlenet(weights=None, aux_logits=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model