"""
Convolutional models.
"""
import torch
import torch.nn as nn

from torchvision import models
from typing import Dict

Tensor = torch.Tensor


class CustomResNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: str = None):
        super(CustomResNet, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
        self.model = model_ft
        if pretrained:
            self.model = torch.load(pretrained)

    def forward(self, x: Dict) -> Tensor:
        z = self.model(x['image']).squeeze()
        return torch.sigmoid(z)
