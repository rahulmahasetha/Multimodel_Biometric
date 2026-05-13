import torch
import numpy as np
import cv2
import os
import sys
from insightface.app import FaceAnalysis

# ================= PATH FIX =================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from transformer_feature_fusion import TransformerFeatureFusion
from fingerprintv2.model_fp import FingerprintNet
from palmprint.model_p import PalmprintNet

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= DATA PATHS =================
DATA_ROOT = os.path.join(ROOT, "Data")
FACE_DIR  = os.path.join(DATA_ROOT, "augmented_face")
FP_DIR    = os.path.join(DATA_ROOT, "fingerprint_augmented")
PALM_DIR  = os.path.join(DATA_ROOT, "palmprint_augmented")

# ================= LOAD MODELS =================
# Face
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Fingerprint
fp_model = FingerprintNet().to(device)
fp_model.load_state_dict(torch.load("../fingerprint/fp_model.pth", map_location=device))
fp_model.eval()

# Palmprint
palm_model = PalmprintNet().to(device)
palm_model.load_state_dict(torch.load("../palmprint/palm_model.pth", map_location=device))
palm_model.eval()

# Fusion
fusion_model = TransformerFeatureFusion().to(device)
fusion_model.load_state_dict(
    torch.load("fusion_transformer.pth", map_location=device)
)
fusion_model.eval()

# ================= HELPERS =================
def get_face_emb(img):
    faces = face_app.get(img)
    if not faces:
        return None
    return torch.tensor(faces[0].normed_embedding).float().to(device)

def get_fp_emb(path):
    img = cv2.imread(path, 0)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128)) / 255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = fp_model(t)[0]
    return e / torch.norm(e)

def get_palm_emb(path):
    img = cv2.imread(path, 0)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128)) / 255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = palm_model(t)[0]
    return e / torch.norm(e)

# ================= BATCH ENROLLMENT =================
fused_db = {}

user_ids = sorted(os.listdir(FACE_DIR))

print(f"\n🔐 Starting batch enrollment for {len(user_ids)} users...\n")

for user_id in user_ids:
    face_user_dir = os.path.join(FACE_DIR, user_id)
    fp_user_dir   = os.path.join(FP_DIR, user_id)
    palm_user_dir = os.path.join(PALM_DIR, user_id)

    if not (os.path.isdir(face_user_dir) and
            os.path.isdir(fp_user_dir) and
            os.path.isdir(palm_user_dir)):
        print(f"⚠️ Skipping user {user_id} (missing modality)")
        continue

    # Take first image from each folder
    face_path = os.path.join(face_user_dir, os.listdir(face_user_dir)[0])
    fp_path   = os.path.join(fp_user_dir, os.listdir(fp_user_dir)[0])
    palm_path = os.path.join(palm_user_dir, os.listdir(palm_user_dir)[0])

    face_img = cv2.imread(face_path)

    face_emb = get_face_emb(face_img)
    fp_emb   = get_fp_emb(fp_path)
    palm_emb = get_palm_emb(palm_path)

    if face_emb is None or fp_emb is None or palm_emb is None:
        print(f"❌ Enrollment failed for user {user_id}")
        continue

    with torch.no_grad():
        fused_emb = fusion_model(
            face_emb.unsqueeze(0),
            fp_emb.unsqueeze(0),
            palm_emb.unsqueeze(0)
        )

    fused_db[user_id] = fused_emb.cpu().numpy()[0]
    print(f"✅ Enrolled user {user_id}")

# ================= SAVE DB =================
np.save("fused_db.npy", fused_db)

print("\n🎉 Batch enrollment completed!")
print(f"Total enrolled users: {len(fused_db)}")
