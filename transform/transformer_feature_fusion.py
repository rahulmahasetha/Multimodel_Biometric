import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerFeatureFusion(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()

        self.face_proj = nn.Linear(512, embed_dim)
        self.fp_proj = nn.Linear(256, embed_dim)
        self.palm_proj = nn.Linear(256, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, face, fp, palm):
        face = self.face_proj(face)
        fp = self.fp_proj(fp)
        palm = self.palm_proj(palm)

        x = torch.stack([face, fp, palm], dim=1)
        x = self.transformer(x)

        x = torch.mean(x, dim=1)
        x = self.fc(x)

        return F.normalize(x, dim=1)