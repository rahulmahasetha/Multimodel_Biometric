import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerFeatureFusion(nn.Module):
    def __init__(
        self,
        face_dim=512,
        fp_dim=256,
        palm_dim=256,
        embed_dim=256,
        num_heads=4,
        num_layers=2
    ):
        super().__init__()

        # Project all modalities to same dimension
        self.face_proj = nn.Linear(face_dim, embed_dim)
        self.fp_proj   = nn.Linear(fp_dim, embed_dim)
        self.palm_proj = nn.Linear(palm_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, face_emb, fp_emb, palm_emb):
        """
        face_emb : (B, 512)
        fp_emb   : (B, 256)
        palm_emb : (B, 256)
        """

        f_face = self.face_proj(face_emb)
        f_fp   = self.fp_proj(fp_emb)
        f_palm = self.palm_proj(palm_emb)

        # Stack as tokens
        x = torch.stack([f_face, f_fp, f_palm], dim=1)  # (B, 3, D)

        x = self.transformer(x)

        # Mean pooling
        fused = x.mean(dim=1)

        fused = self.fc(fused)
        fused = F.normalize(fused, p=2, dim=1)

        return fused
