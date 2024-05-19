from torch import nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, mnasnet1_3, MNASNet1_3_Weights


def get_mobilenet_model():
    mobilenet_model = mobilenet_v3_large(MobileNet_V3_Large_Weights.DEFAULT)
    mobilenet_model.classifier = nn.Sequential(*list(mobilenet_model.classifier.children())[:-3])
    return mobilenet_model


def get_mnasnet_model():
    mnasnet1_3_model = mnasnet1_3(MNASNet1_3_Weights.DEFAULT)
    mnasnet1_3_model.classifier[1] = nn.Flatten()
    return mnasnet1_3_model
