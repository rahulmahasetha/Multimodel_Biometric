import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class FingerprintNet(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        base = models.resnet18(pretrained=True)

        # Change first layer for grayscale fingerprint
        base.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        base.fc = nn.Identity()
        self.backbone = base

        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)

'''import torch.nn as nn
import torch.nn.functional as F

class FingerprintNet(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        # EXACTLY match the checkpoint keys:
        # cnn.0, cnn.3, cnn.6
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # cnn.0
            nn.ReLU(),                                    # cnn.1
            nn.Identity(),                                # cnn.2 (placeholder)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # cnn.3
            nn.ReLU(),                                    # cnn.4
            nn.Identity(),                                # cnn.5 (placeholder)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # cnn.6
            nn.ReLU(),                                    # cnn.7

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Matches fc.weight shape [256, 128]
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)   # (B, 128)
        x = self.fc(x)
        return F.normalize(x, dim=1)
'''
''''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class FingerprintNet(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        # ResNet backbone (matches checkpoint keys: backbone.*)
        backbone = resnet18(weights=None)

        # Change first conv layer for grayscale input (1 channel)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove final classification layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Embedding layer (matches embedding.* keys)
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return F.normalize(x, dim=1)'''