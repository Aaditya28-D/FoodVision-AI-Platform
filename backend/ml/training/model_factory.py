from ml.models.efficientnet import build_efficientnet_b0
from ml.models.googlenet import build_googlenet
from ml.models.mobilenet import build_mobilenet_v3_large
from ml.models.resnet import build_resnet50
from ml.models.vit import build_vit_b_16


def build_model(
    model_name: str,
    num_classes: int,
    use_pretrained: bool = True,
):
    if model_name == "efficientnet_b0":
        return build_efficientnet_b0(num_classes=num_classes, use_pretrained=use_pretrained)

    if model_name == "resnet50":
        return build_resnet50(num_classes=num_classes, use_pretrained=use_pretrained)

    if model_name == "mobilenet_v3_large":
        return build_mobilenet_v3_large(num_classes=num_classes, use_pretrained=use_pretrained)

    if model_name == "googlenet":
        return build_googlenet(num_classes=num_classes, use_pretrained=use_pretrained)

    if model_name == "vit_b_16":
        return build_vit_b_16(num_classes=num_classes, use_pretrained=use_pretrained)

    raise ValueError(f"Unsupported model_name: {model_name}")