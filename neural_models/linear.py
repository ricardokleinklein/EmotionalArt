"""
Linear models
"""

import torch
import torch.nn as nn

from typing import Dict


class SimpleClassifier(nn.Module):

    def __init__(self, in_features: int, num_classes: int) -> None:
        super(SimpleClassifier, self).__init__()
        self.cls = nn.Linear(in_features=in_features,
                             out_features=num_classes)

    def forward(self, x: Dict) -> torch.Tensor:
        z = self.cls(x['vector'])
        return nn.functional.log_softmax(z, dim=1)
