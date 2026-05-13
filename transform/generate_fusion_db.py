import os
import sys
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

# ================= PATH =================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from fingerprint.model_fp import FingerprintNet
from palmprint.model_p import PalmprintNet
from transform.transformer_feature_fusion import TransformerFeatureFusion

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= DATA PATH =================
DATA_ROOT = os.path.join(ROOT, "Data")
FACE_DIR = os.path.join(DATA_ROOT, "augmented_face")
FP_DIR   = os.path.join(DATA_ROOT, "fingerprint_augmented")
PALM_DIR = os.path.join(DATA_ROOT, "palmprint_augmented")

USERS = sorted(os.listdir(FACE_DIR))

# ================= LOAD MODELS =================
print("Loading models...")

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

fp_model = FingerprintNet().to(device)
fp_model.load_state_dict(
    torch.load(os.path.join(ROOT, "fingerprint", "fp_model.pth"),
               map_location=device)
)
fp_model.eval()

palm_model = PalmprintNet().to(device)
palm_model.load_state_dict(
    torch.load(os.path.join(ROOT, "palmprint", "palm_model.pth"),
               map_location=device)
)
palm_model.eval()

fusion_model = TransformerFeatureFusion().to(device)
fusion_model.load_state_dict(
    torch.load(os.path.join(ROOT, "transform", "fusion_transformer.pth"),
               map_location=device)
)
fusion_model.eval()

print("Models loaded successfully.\n")

# ================= HELPER FUNCTIONS =================
def get_face_emb(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_app.get(img)
    if not faces:
        return None
    return torch.tensor(faces[0].normed_embedding).float().to(device)

def get_fp_emb(path):
    img = cv2.imread(path, 0)
    if img is None:
        return None
    img = cv2.resize(img, (128,128)) / 255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = fp_model(t)[0]
    return e / torch.norm(e)

def get_palm_emb(path):
    img = cv2.imread(path, 0)
    if img is None:
        return None
    img = cv2.resize(img, (128,128)) / 255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = palm_model(t)[0]
    return e / torch.norm(e)

# ================= GENERATE FUSED DB =================
print("Generating fused database...")

fused_db = {}

for uid in USERS:

    face_path = os.path.join(FACE_DIR, uid)
    fp_path   = os.path.join(FP_DIR, uid)
    palm_path = os.path.join(PALM_DIR, uid)

    if not (os.path.isdir(face_path) and
            os.path.isdir(fp_path) and
            os.path.isdir(palm_path)):
        continue

    files = sorted(os.listdir(face_path))
    if len(files) == 0:
        continue

    # Use first image as enrollment
    face_img  = os.path.join(face_path, files[0])
    fp_img    = os.path.join(fp_path, files[0])
    palm_img  = os.path.join(palm_path, files[0])

    face_emb = get_face_emb(face_img)
    fp_emb   = get_fp_emb(fp_img)
    palm_emb = get_palm_emb(palm_img)

    if face_emb is None or fp_emb is None or palm_emb is None:
        continue

    with torch.no_grad():
        fused = fusion_model(
            face_emb.unsqueeze(0),
            fp_emb.unsqueeze(0),
            palm_emb.unsqueeze(0)
        )[0]

    fused_db[uid] = fused.cpu().numpy()

print("Total enrolled users:", len(fused_db))

# ================= SAVE FILE =================
save_path = os.path.join(ROOT, "transform", "fused_db.npy")

np.save(save_path, fused_db)

print("\nFused DB saved at:")
print(save_path)
print("Done.")