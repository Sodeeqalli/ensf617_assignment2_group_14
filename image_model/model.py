import torch.nn as nn
import torchvision.models as models


def build_model(num_classes=4):
    # Load pretrained MobileNetV2 backbone (ImageNet weights)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Replace the final classifier layer to match our dataset classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),                 # regularization
        nn.Linear(in_features, num_classes),  # output layer for class prediction
    )

    return model