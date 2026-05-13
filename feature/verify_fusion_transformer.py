'''import torch
import numpy as np
from transformer_feature_fusion import TransformerFeatureFusion

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

fusion_model = TransformerFeatureFusion().to(device)
fusion_model.load_state_dict(
    torch.load("fusion_transformer.pth", map_location=device)
)
fusion_model.eval()

def verify_fusion_transformer(face_emb, fp_emb, palm_emb, fused_db):
    with torch.no_grad():
        fused_query = fusion_model(
            face_emb.unsqueeze(0),
            fp_emb.unsqueeze(0),
            palm_emb.unsqueeze(0)
        )

    best_id = None
    best_score = -1.0

    for pid, ref_emb in fused_db.items():
        ref_emb = torch.tensor(ref_emb).float().to(device).unsqueeze(0)
        score = torch.cosine_similarity(fused_query, ref_emb).item()

        if score > best_score:
            best_score = score
            best_id = pid

    return best_id, best_score
'''
import os
import sys
import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from feature.transformer_feature_fusion import TransformerFeatureFusion

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

fusion_model = TransformerFeatureFusion().to(device)
fusion_model.load_state_dict(
    torch.load(os.path.join(ROOT, "feature", "fusion_transformer.pth"),
               map_location=device)
)
fusion_model.eval()


def verify_fusion_transformer(face_emb, fp_emb, palm_emb, fused_db):
    """
    Transformer feature fusion with consistency gating
    """

    with torch.no_grad():
        fused_query = fusion_model(
            face_emb.unsqueeze(0),
            fp_emb.unsqueeze(0),
            palm_emb.unsqueeze(0)
        )

    best_id = None
    best_score = -1.0

    for user_id, ref_fused in fused_db.items():
        ref_fused = torch.tensor(ref_fused).float().to(device).unsqueeze(0)
        score = torch.cosine_similarity(fused_query, ref_fused).item()

        if score > best_score:
            best_score = score
            best_id = user_id

    return best_id, best_score
