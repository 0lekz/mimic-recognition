import torch.nn as nn


class MyCNN(nn.Module):
    """
    Mid-size CNN for 48x48 grayscale facial expression classification 3 output classes:
    Happy, Sad, Neutral
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # -- Conv Blocks --
        # Hierarchical feature learning
        # conv block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 48x48 -> 24x24
            nn.Dropout(0.05),
        )

        # conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24x24 -> 12x12
            nn.Dropout(0.2),
        )

        # conv block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12x12 -> 6x6
            nn.Dropout(0.25),
        )

        # adaptive pooling to reduce to (1x1) features per channel
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    # forward pass
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x
