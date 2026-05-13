'''import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from insightface.app import FaceAnalysis
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from fingerprintv2.model_fp import FingerprintNet
from palmprint.model_p import PalmprintNet
from feature.transformer_feature_fusion import TransformerFeatureFusion

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= PATHS =================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(ROOT, "Data")

FACE_DIR = os.path.join(DATA_ROOT, "augmented_face")
FP_DIR   = os.path.join(DATA_ROOT, "fingerprint_augmented")
PALM_DIR = os.path.join(DATA_ROOT, "palmprint_augmented")

USERS = [str(i) for i in range(1, 150)]   # 👈 149 users only

# ================= LOAD MODELS =================
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

fp_model = FingerprintNet().to(device)
FP_MODEL_PATH = os.path.join(ROOT, "fingerprintv2", "fp_model.pth")
fp_model.load_state_dict(torch.load(FP_MODEL_PATH, map_location=device))
fp_model.eval()

palm_model = PalmprintNet().to(device)
PALM_MODEL_PATH = os.path.join(ROOT, "palmprint", "palm_model.pth")
palm_model.load_state_dict(torch.load(PALM_MODEL_PATH, map_location=device))
#palm_model.load_state_dict(torch.load("../palmprint/palm_model.pth", map_location=device))
palm_model.eval()

fusion_model = TransformerFeatureFusion().to(device)
FUSION_MODEL_PATH = os.path.join(ROOT, "fusion_transformer.pth")
fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=device))
#fusion_model.load_state_dict(torch.load("fusion_transformer.pth", map_location=device))
fusion_model.eval()

# ================= HELPERS =================
def get_face_emb(path):
    img = cv2.imread(path)
    faces = face_app.get(img)
    if not faces:
        return None
    return torch.tensor(faces[0].normed_embedding).float().to(device)

def get_fp_emb(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (128,128)) / 255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = fp_model(t)[0]
    return e / torch.norm(e)

def get_palm_emb(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (128,128)) / 255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = palm_model(t)[0]
    return e / torch.norm(e)

def fuse(face, fp, palm):
    with torch.no_grad():
        return fusion_model(
            face.unsqueeze(0),
            fp.unsqueeze(0),
            palm.unsqueeze(0)
        )[0]

def cosine(a, b):
    return torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# ================= BUILD ENROLLMENT TEMPLATES =================
print("\n📌 Building enrollment templates...")
enroll_db = {}

for uid in USERS:
    fdir = os.path.join(FACE_DIR, uid)
    fpdir = os.path.join(FP_DIR, uid)
    pdir = os.path.join(PALM_DIR, uid)

    if not (os.path.isdir(fdir) and os.path.isdir(fpdir) and os.path.isdir(pdir)):
        continue

    face_img = sorted(os.listdir(fdir))[0]   # 👈 image 1 = ENROLL
    fp_img   = sorted(os.listdir(fpdir))[0]
    palm_img = sorted(os.listdir(pdir))[0]

    fe = get_face_emb(os.path.join(fdir, face_img))
    fpe = get_fp_emb(os.path.join(fpdir, fp_img))
    pe = get_palm_emb(os.path.join(pdir, palm_img))

    if fe is None or fpe is None or pe is None:
        continue

    enroll_db[uid] = fuse(fe, fpe, pe)

# ================= EVALUATION =================
genuine_scores = []
impostor_scores = []

print("\n🔍 Running correct fusion evaluation...")

for uid in tqdm(enroll_db.keys()):
    fdir = os.path.join(FACE_DIR, uid)
    fpdir = os.path.join(FP_DIR, uid)
    pdir = os.path.join(PALM_DIR, uid)

    files_f = sorted(os.listdir(fdir))
    files_fp = sorted(os.listdir(fpdir))
    files_p = sorted(os.listdir(pdir))

    if len(files_f) < 2:
        continue

    # 👈 image 2 = PROBE
    fe = get_face_emb(os.path.join(fdir, files_f[1]))
    fpe = get_fp_emb(os.path.join(fpdir, files_fp[1]))
    pe = get_palm_emb(os.path.join(pdir, files_p[1]))

    probe = fuse(fe, fpe, pe)

    # ---------- Genuine ----------
    g = cosine(probe, enroll_db[uid])
    genuine_scores.append(g)

    # ---------- Impostor ----------
    for other in enroll_db:
        if other == uid:
            continue
        i = cosine(probe, enroll_db[other])
        impostor_scores.append(i)

# ================= METRICS =================
TH = 0.65

FAR = sum(s >= TH for s in impostor_scores) / len(impostor_scores)
FRR = sum(s < TH for s in genuine_scores) / len(genuine_scores)

accuracy = (
    (len(genuine_scores) - sum(s < TH for s in genuine_scores)) +
    (len(impostor_scores) - sum(s >= TH for s in impostor_scores))
) / (len(genuine_scores) + len(impostor_scores))

# ================= REPORT =================
print("\n📊 CORRECT FUSION EVALUATION REPORT")
print("======================================")
print(f"Users Evaluated     : {len(enroll_db)}")
print(f"Genuine Comparisons : {len(genuine_scores)}")
print(f"Impostor Comparisons: {len(impostor_scores)}")
print("--------------------------------------")
print(f"Threshold           : {TH}")
print(f"Accuracy            : {accuracy*100:.2f}%")
print(f"FAR                 : {FAR*100:.2f}%")
print(f"FRR                 : {FRR*100:.2f}%")
print("======================================")
'''
''''
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import cv2

import numpy as np
import torch
from tqdm import tqdm
from insightface.app import FaceAnalysis
from fingerprint.model_fp import FingerprintNet
from palmprint.model_p import PalmprintNet
from feature.transformer_feature_fusion import TransformerFeatureFusion

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= PATHS =================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(ROOT, "Data")

FACE_DIR = os.path.join(DATA_ROOT, "augmented_face")
FP_DIR   = os.path.join(DATA_ROOT, "fingerprint_augmented")
PALM_DIR = os.path.join(DATA_ROOT, "palmprint_augmented")

USERS = [str(i) for i in range(1, 150)]

# ================= LOAD MODELS =================
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

fp_model = FingerprintNet().to(device)
fp_model.load_state_dict(torch.load(os.path.join(ROOT,"fingerprint","fp_model.pth"), map_location=device))
fp_model.eval()

palm_model = PalmprintNet().to(device)
palm_model.load_state_dict(torch.load(os.path.join(ROOT,"palmprint","palm_model.pth"), map_location=device))
palm_model.eval()

fusion_model = TransformerFeatureFusion().to(device)
torch.load(os.path.join(ROOT,"feature","fusion_transformer.pth"),map_location=device)
fusion_model.eval()

# ================= HELPERS =================
def get_face_emb(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_app.get(img)
    if not faces:
        return None
    return torch.tensor(faces[0].normed_embedding).float().to(device)

def get_fp_emb(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (128,128)) / 255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = fp_model(t)[0]
    return e / torch.norm(e)

def get_palm_emb(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (128,128)) / 255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = palm_model(t)[0]
    return e / torch.norm(e)

def fuse(face, fp, palm):
    with torch.no_grad():
        return fusion_model(
            face.unsqueeze(0),
            fp.unsqueeze(0),
            palm.unsqueeze(0)
        )[0]

def cosine(a, b):
    return torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# ================= BUILD ENROLLMENT =================
print("\nBuilding enrollment templates...")
enroll_db = {}

for uid in USERS:
    fdir = os.path.join(FACE_DIR, uid)
    fpdir = os.path.join(FP_DIR, uid)
    pdir = os.path.join(PALM_DIR, uid)

    if not os.path.isdir(fdir): continue

    fe = get_face_emb(os.path.join(fdir, sorted(os.listdir(fdir))[0]))
    fpe = get_fp_emb(os.path.join(fpdir, sorted(os.listdir(fpdir))[0]))
    pe = get_palm_emb(os.path.join(pdir, sorted(os.listdir(pdir))[0]))

    if fe is None: continue

    enroll_db[uid] = fuse(fe, fpe, pe)

# ================= EVALUATION =================
genuine_scores = []
impostor_scores = []
correct = 0
total = 0

print("\nRunning fusion evaluation...")

for uid in tqdm(enroll_db.keys()):
    fdir = os.path.join(FACE_DIR, uid)
    fpdir = os.path.join(FP_DIR, uid)
    pdir = os.path.join(PALM_DIR, uid)

    files_f = sorted(os.listdir(fdir))
    files_fp = sorted(os.listdir(fpdir))
    files_p = sorted(os.listdir(pdir))

    if len(files_f) < 2: continue

    fe = get_face_emb(os.path.join(fdir, files_f[1]))
    fpe = get_fp_emb(os.path.join(fpdir, files_fp[1]))
    pe = get_palm_emb(os.path.join(pdir, files_p[1]))

    probe = fuse(fe, fpe, pe)

    # Compare with ALL enrolled templates
    scores = {u: cosine(probe, enroll_db[u]) for u in enroll_db}

    predicted_user = max(scores, key=scores.get)
    best_score = scores[predicted_user]

    total += 1

    if predicted_user == uid:
        genuine_scores.append(best_score)
        correct += 1
    else:
        impostor_scores.append(best_score)

# ================= METRICS =================
TH = 0.7

FAR = sum(s >= TH for s in impostor_scores) / len(impostor_scores) if impostor_scores else 0
FRR = sum(s < TH for s in genuine_scores) / len(genuine_scores) if genuine_scores else 0
accuracy = correct / total

print("\n===== FINAL REPORT =====")
print("Users Evaluated :", len(enroll_db))
print("Accuracy        :", round(accuracy*100,2), "%")
print("FAR             :", round(FAR*100,2), "%")
print("FRR             :", round(FRR*100,2), "%")
print("========================")'''

import os
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis

# ================= PATH SETUP =================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from fingerprint.model_fp import FingerprintNet
from palmprint.model_p import PalmprintNet
from transform.transformer_feature_fusion import TransformerFeatureFusion

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= DATA PATHS =================
DATA_ROOT = os.path.join(ROOT, "Data")

FACE_DIR = os.path.join(DATA_ROOT, "augmented_face")
FP_DIR   = os.path.join(DATA_ROOT, "fingerprint_augmented")
PALM_DIR = os.path.join(DATA_ROOT, "palmprint_augmented")

USERS = [str(i) for i in range(1, 150)]

# ================= LOAD MODELS =================
print("\nLoading models...")

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

fp_model = FingerprintNet().to(device)
fp_model.load_state_dict(
    torch.load(os.path.join(ROOT,"fingerprint","fp_model.pth"),
               map_location=device)
)
fp_model.eval()

palm_model = PalmprintNet().to(device)
palm_model.load_state_dict(
    torch.load(os.path.join(ROOT,"palmprint","palm_model.pth"),
               map_location=device)
)
palm_model.eval()

fusion_model = TransformerFeatureFusion().to(device)
fusion_model.load_state_dict(
    torch.load(os.path.join(ROOT,"transform","fusion_transformer.pth"),
               map_location=device)
)
fusion_model.eval()

print("Models loaded successfully.")

# ================= HELPER FUNCTIONS =================
def get_face_emb(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_app.get(img)
    if not faces:
        return None
    return torch.tensor(faces[0].normed_embedding).float().to(device)

def get_fp_emb(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (128,128)) / 255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = fp_model(t)[0]
    return e / torch.norm(e)

def get_palm_emb(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (128,128)) / 255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = palm_model(t)[0]
    return e / torch.norm(e)

def fuse(face, fp, palm):
    with torch.no_grad():
        return fusion_model(
            face.unsqueeze(0),
            fp.unsqueeze(0),
            palm.unsqueeze(0)
        )[0]

def cosine(a, b):
    return torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# ================= BUILD ENROLLMENT =================
print("\nBuilding enrollment templates...")
enroll_db = {}

for uid in USERS:
    fdir = os.path.join(FACE_DIR, uid)
    fpdir = os.path.join(FP_DIR, uid)
    pdir = os.path.join(PALM_DIR, uid)

    if not os.path.isdir(fdir):
        continue

    files_f = sorted(os.listdir(fdir))
    files_fp = sorted(os.listdir(fpdir))
    files_p = sorted(os.listdir(pdir))

    if len(files_f) == 0:
        continue

    fe = get_face_emb(os.path.join(fdir, files_f[0]))
    fpe = get_fp_emb(os.path.join(fpdir, files_fp[0]))
    pe = get_palm_emb(os.path.join(pdir, files_p[0]))

    if fe is None:
        continue

    enroll_db[uid] = fuse(fe, fpe, pe)

print("Enrollment complete:", len(enroll_db), "users")

# ================= SANITY TEST =================
import random

users = list(enroll_db.keys())

if len(users) >= 2:
    u1, u2 = random.sample(users, 2)

    sim = cosine(enroll_db[u1], enroll_db[u2])

    print("\n🔍 Sanity Test (Random Identity Similarity)")
    print(f"User {u1} vs User {u2} similarity: {sim:.4f}")
# ================= VERIFICATION EVALUATION =================
genuine_scores = []
impostor_scores = []

print("\nRunning research-grade fusion evaluation...")

for uid in tqdm(enroll_db.keys()):
    fdir = os.path.join(FACE_DIR, uid)
    fpdir = os.path.join(FP_DIR, uid)
    pdir = os.path.join(PALM_DIR, uid)

    files_f = sorted(os.listdir(fdir))
    files_fp = sorted(os.listdir(fpdir))
    files_p = sorted(os.listdir(pdir))

    if len(files_f) < 2:
        continue

    fe = get_face_emb(os.path.join(fdir, files_f[1]))
    fpe = get_fp_emb(os.path.join(fpdir, files_fp[1]))
    pe = get_palm_emb(os.path.join(pdir, files_p[1]))

    if fe is None:
        continue

    probe = fuse(fe, fpe, pe)

    for other_uid in enroll_db:
        score = cosine(probe, enroll_db[other_uid])

        if other_uid == uid:
            genuine_scores.append(score)
        else:
            impostor_scores.append(score)

# ================= METRICS =================
genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)

TH = 0.7

FAR = np.sum(impostor_scores >= TH) / len(impostor_scores)
FRR = np.sum(genuine_scores < TH) / len(genuine_scores)

accuracy = (
    np.sum(genuine_scores >= TH) +
    np.sum(impostor_scores < TH)
) / (len(genuine_scores) + len(impostor_scores))

print("\n===== VERIFICATION REPORT =====")
print("Users Evaluated :", len(enroll_db))
print("Genuine mean:", np.mean(genuine_scores))
print("Impostor mean:", np.mean(impostor_scores))
#print("Genuine Trials  :", len(genuine_scores))
#print("Impostor Trials :", len(impostor_scores))
print("--------------------------------")
print("Threshold       :", TH)
print("Accuracy        :", round(accuracy*100,2), "%")
print("FAR             :", round(FAR*100,2), "%")
print("FRR             :", round(FRR*100,2), "%")

# ================= EER =================
all_scores = np.concatenate([genuine_scores, impostor_scores])
thresholds = np.linspace(min(all_scores), max(all_scores), 1000)

best_diff = 1
eer = 0
eer_th = 0

for th in thresholds:
    far = np.sum(impostor_scores >= th) / len(impostor_scores)
    frr = np.sum(genuine_scores < th) / len(genuine_scores)

    if abs(far - frr) < best_diff:
        best_diff = abs(far - frr)
        eer = (far + frr) / 2
        eer_th = th

print("EER              :", round(eer*100,3), "%")
print("EER Threshold    :", round(eer_th,3))
print("================================")

# ================= ROC CURVE =================
fars = []
tars = []

for th in thresholds:
    far = np.sum(impostor_scores >= th) / len(impostor_scores)
    tar = np.sum(genuine_scores >= th) / len(genuine_scores)

    fars.append(far)
    tars.append(tar)

plt.figure()
plt.plot(fars, tars)
plt.xlabel("False Acceptance Rate")
plt.ylabel("True Acceptance Rate")
plt.title("ROC Curve - Transformer Fusion")
plt.grid()
plt.show()