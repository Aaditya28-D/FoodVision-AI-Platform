from enum import Enum


class ModelName(str, Enum):
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    EFFICIENTNET_B0 = "efficientnet_b0"
    RESNET50 = "resnet50"
    GOOGLENET = "googlenet"