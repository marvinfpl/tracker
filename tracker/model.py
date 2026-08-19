import torch.nn as nn
import os

class Tracker(nn.Module):
    def __init__(self):
        super.__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, (2, 1)),
            nn.MaxPool2d(5, 1),
            nn.Conv2d(16, 32, 5, (2, 1)),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(32, 128, 3, (2, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)