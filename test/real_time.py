import cv2
import os
import sys
import numpy as np
import torch
from insightface.app import FaceAnalysis

# ================= PATH FIX =================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ================= IMPORTS =================
from fingerprint.model_fp import FingerprintNet
from palmprint.model_p import PalmprintNet
from transform.transformer_feature_fusion import TransformerFeatureFusion

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= LOAD DATABASES =================
face_db = np.load(os.path.join(ROOT, "templates", "face_db.npy"),
                  allow_pickle=True).item()
fp_db = np.load(os.path.join(ROOT, "fingerprint", "fp_db.npy"),
                allow_pickle=True).item()
palm_db = np.load(os.path.join(ROOT, "palmprint", "palm_db.npy"),
                  allow_pickle=True).item()
fused_db_np = np.load(os.path.join(ROOT, "transform", "fused_db.npy"),
                      allow_pickle=True).item()

# Convert fused DB to tensors
fused_db = {
    k: torch.tensor(v).float().to(device)
    for k, v in fused_db_np.items()
}

# ================= LOAD MODELS =================
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

# ================= HELPER =================
def load_gray(path):
    if os.path.isdir(path):
        files = [f for f in os.listdir(path)
                 if f.lower().endswith((".jpg",".png",".jpeg"))]
        if not files:
            return None
        path = os.path.join(path, files[0])
    return cv2.imread(path, 0)

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
    img = load_gray(path)
    if img is None:
        return None
    img = cv2.resize(img,(128,128))/255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        emb = fp_model(t)[0]
    return emb / torch.norm(emb)

def get_palm_emb(path):
    img = load_gray(path)
    if img is None:
        return None
    img = cv2.resize(img,(128,128))/255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        emb = palm_model(t)[0]
    return emb / torch.norm(emb)

# ================= MAIN =================
print("\n🔐 STRICT MULTIMODAL VERIFICATION 🔐\n")

face_path = input("Enter face image path: ")
fp_path   = input("Enter fingerprint image path: ")
palm_path = input("Enter palmprint image path: ")

face_emb = get_face_emb(face_path)
fp_emb   = get_fp_emb(fp_path)
palm_emb = get_palm_emb(palm_path)

if face_emb is None or fp_emb is None or palm_emb is None:
    print("❌ Biometric extraction failed.")
    sys.exit(0)

# ================= THRESHOLDS =================
FACE_TH   = 0.75
FP_TH     = 0.75
PALM_TH   = 0.75
FUSION_TH = 0.91  # from research

best_user = None
best_score = -1.0

# ================= MATCHING =================
for user_id in face_db.keys():

    face_ref = torch.tensor(face_db[user_id]).float().to(device)
    fp_ref   = torch.tensor(fp_db[user_id]).float().to(device)
    palm_ref = torch.tensor(palm_db[user_id]).float().to(device)

    face_sim = torch.cosine_similarity(
        face_emb.unsqueeze(0), face_ref.unsqueeze(0)
    ).item()

    fp_sim = torch.cosine_similarity(
        fp_emb.unsqueeze(0), fp_ref.unsqueeze(0)
    ).item()

    palm_sim = torch.cosine_similarity(
        palm_emb.unsqueeze(0), palm_ref.unsqueeze(0)
    ).item()

    # 🔒 Strict gating: all 3 must match
    if not (face_sim >= FACE_TH and
            fp_sim >= FP_TH and
            palm_sim >= PALM_TH):
        continue

    # Compute fusion once user passes all 3
    with torch.no_grad():
        fused_query = fusion_model(
            face_emb.unsqueeze(0),
            fp_emb.unsqueeze(0),
            palm_emb.unsqueeze(0)
        )

    fused_ref = fused_db[user_id].unsqueeze(0)

    fusion_sim = torch.cosine_similarity(
        fused_query, fused_ref
    ).item()

    if fusion_sim >= FUSION_TH and fusion_sim > best_score:
        best_score = fusion_sim
        best_user = user_id

# ================= RESULT =================
print("\n================ RESULT ================")

if best_user is not None:
    print("Matched User :", best_user)
    print("Fusion Score :", round(best_score,4))
    print("✅ ACCESS GRANTED")
else:
    print("❌ ACCESS DENIED")

print("=======================================\n")


'''import cv2
import os
import sys
import numpy as np
import torch
from insightface.app import FaceAnalysis

# ================= PATH FIX =================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ================= IMPORTS =================
from fingerprint.model_fp import FingerprintNet
from palmprint.model_p import PalmprintNet
from feature.verify_fusion_transformer import fusion_model  # only the model

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= LOAD DATABASES =================
# Individual modality templates (EMBEDDINGS ONLY)
face_db = np.load(os.path.join(ROOT, "templates", "face_db.npy"), allow_pickle=True).item()
fp_db   = np.load(os.path.join(ROOT, "fingerprint", "fp_db.npy"), allow_pickle=True).item()
palm_db = np.load(os.path.join(ROOT, "palmprint", "palm_db.npy"), allow_pickle=True).item()

# Fused templates (Transformer output)
fused_db = np.load(os.path.join(ROOT, "feature", "fused_db.npy"), allow_pickle=True).item()

# ================= LOAD MODELS =================
# Face (ArcFace)
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Fingerprint
fp_model = FingerprintNet().to(device)
fp_model.load_state_dict(
    torch.load(os.path.join(ROOT, "fingerprint", "fp_model.pth"),
               map_location=device)
)
fp_model.eval()

# Palmprint
palm_model = PalmprintNet().to(device)
palm_model.load_state_dict(
    torch.load(os.path.join(ROOT, "palmprint", "palm_model.pth"),
               map_location=device)
)
palm_model.eval()

# ================= HELPER FUNCTIONS =================
def load_image(path):
    if os.path.isdir(path):
        files = [f for f in os.listdir(path)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if not files:
            return None
        path = os.path.join(path, files[0])
        print(f"📂 Using image: {path}")
    return cv2.imread(path, 0)

def get_face_emb(img):
    faces = face_app.get(img)
    if not faces:
        return None
    return torch.tensor(faces[0].normed_embedding).float().to(device)

def get_fp_emb(path):
    img = load_image(path)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128)) / 255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = fp_model(t)[0]
    return e / torch.norm(e)

def get_palm_emb(path):
    img = load_image(path)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128)) / 255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = palm_model(t)[0]
    return e / torch.norm(e)

# ================= MAIN =================
print("\n🔐 STRICT MULTIMODAL VERIFICATION (EMBEDDING-ONLY) 🔐")

face_path = input("Enter face image path: ")
fp_path   = input("Enter fingerprint image path (file/folder): ")
palm_path = input("Enter palmprint image path (file/folder): ")

face_img = cv2.imread(face_path)

face_emb = get_face_emb(face_img)
fp_emb   = get_fp_emb(fp_path)
palm_emb = get_palm_emb(palm_path)

if face_emb is None or fp_emb is None or palm_emb is None:
    print("❌ Biometric extraction failed")
    sys.exit(0)

# ================= STRICT VERIFICATION =================
FACE_TH = 0.8
FP_TH   = 0.8
PALM_TH = 0.8
FUSION_TH = 0.8

best_user = None
best_fusion_score = -1.0

for user_id in face_db.keys():

    # --- Load enrolled embeddings ---
    face_ref = torch.tensor(face_db[user_id]).float().to(device)
    fp_ref   = torch.tensor(fp_db[user_id]).float().to(device)
    palm_ref = torch.tensor(palm_db[user_id]).float().to(device)

    # --- Per-modality similarity ---
    face_sim = torch.cosine_similarity(face_emb.unsqueeze(0),
                                       face_ref.unsqueeze(0)).item()
    fp_sim   = torch.cosine_similarity(fp_emb.unsqueeze(0),
                                       fp_ref.unsqueeze(0)).item()
    palm_sim = torch.cosine_similarity(palm_emb.unsqueeze(0),
                                       palm_ref.unsqueeze(0)).item()

    # ❌ reject if ANY modality mismatches
    if not (face_sim >= FACE_TH and fp_sim >= FP_TH and palm_sim >= PALM_TH):
        continue

    # ✅ SAME PERSON confirmed → now apply fusion
    with torch.no_grad():
        fused_query = fusion_model(
            face_emb.unsqueeze(0),
            fp_emb.unsqueeze(0),
            palm_emb.unsqueeze(0)
        )

    fused_ref = torch.tensor(fused_db[user_id]).float().to(device).unsqueeze(0)
    fusion_sim = torch.cosine_similarity(fused_query, fused_ref).item()

    if fusion_sim >= FUSION_TH and fusion_sim > best_fusion_score:
        best_fusion_score = fusion_sim
        best_user = user_id

# ================= RESULT =================
print("\n================ RESULT ================")

if best_user is not None:
    print(f"Matched User      : {best_user}")
    print(f"Fusion Similarity : {best_fusion_score:.3f}")
    print("✅ ACCESS GRANTED (ALL MODALITIES MATCH)")
else:
    print("❌ ACCESS DENIED (Cross-person biometric mismatch)")

print("=======================================\n")
'''