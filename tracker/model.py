import torch.nn as nn
import os

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, (2, 1)),
    nn.MaxPool2d(5, 1),
    nn.Conv2d(16, 32, 5, (2, 1)),
    nn.MaxPool2d(3, 2),
    nn.Conv2d(32, 128, 3, (2, 1)),
    nn.Flatten(start_dim=(os.getenv('batch_size'), 128, )),
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 2)
)