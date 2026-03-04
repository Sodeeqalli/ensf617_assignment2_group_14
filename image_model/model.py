import torch.nn as nn
import torchvision.models as models


def build_model(num_classes=4):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )
    return model