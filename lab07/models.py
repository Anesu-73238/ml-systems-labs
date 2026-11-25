import torch.nn as nn


class CNN(nn.Module):
    """Same layout as BaselineCNN, with configurable channel counts and hidden size."""

    def __init__(self, channels=(32, 64, 128, 256), hidden=256):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, 3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(c4 * 3 * 3, hidden), nn.ReLU(), nn.Linear(hidden, 10))

    def forward(self, x):
        return self.classifier(self.features(x))