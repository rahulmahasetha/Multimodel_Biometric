import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class PalmprintNet(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        base = models.resnet18(pretrained=True)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
