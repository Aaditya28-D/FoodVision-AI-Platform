from enum import Enum


class ModelName(str, Enum):
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    GOOGLENET = "googlenet"