from torchvision import models
from torch import nn


def build_vit_b_16(num_classes: int) -> nn.Module:
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model