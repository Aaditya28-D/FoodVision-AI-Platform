from torch import nn

from ml.models.efficientnet import build_efficientnet_b0
from ml.models.googlenet import build_googlenet
from ml.models.mobilenet import build_mobilenet_v3_large
from ml.models.resnet import build_resnet50


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "mobilenet_v3_large":
        return build_mobilenet_v3_large(num_classes=num_classes)

    if model_name == "efficientnet_b0":
        return build_efficientnet_b0(num_classes=num_classes)

    if model_name == "resnet50":
        return build_resnet50(num_classes=num_classes)

    if model_name == "googlenet":
        return build_googlenet(num_classes=num_classes)

    raise ValueError(f"Unsupported model name: {model_name}")