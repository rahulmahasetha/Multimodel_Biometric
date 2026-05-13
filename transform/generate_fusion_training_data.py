import os
import sys
import cv2
import torch
from tqdm import tqdm
from insightface.app import FaceAnalysis

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from fingerprint.model_fp import FingerprintNet
from palmprint.model_p import PalmprintNet

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_ROOT = os.path.join(ROOT, "Data")

FACE_DIR = os.path.join(DATA_ROOT, "augmented_face")
FP_DIR   = os.path.join(DATA_ROOT, "fingerprint_augmented")
PALM_DIR = os.path.join(DATA_ROOT, "palmprint_augmented")

# --------- LOAD USER IDS SAFELY ----------
USERS = sorted(
    [u for u in os.listdir(FACE_DIR) if u.isdigit()],
    key=lambda x: int(x)
)

print("Total detected users:", len(USERS))

# --------- LOAD MODELS ----------
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

fp_model = FingerprintNet().to(device)
fp_model.load_state_dict(
    torch.load(os.path.join(ROOT, "fingerprint", "fp_model.pth"), map_location=device)
)
fp_model.eval()

palm_model = PalmprintNet().to(device)
palm_model.load_state_dict(
    torch.load(os.path.join(ROOT, "palmprint", "palm_model.pth"), map_location=device)
)
palm_model.eval()

face_list = []
fp_list = []
palm_list = []
label_list = []

print("Generating fusion training data...")

for label, uid in enumerate(tqdm(USERS)):

    face_path = os.path.join(FACE_DIR, uid)
    fp_path   = os.path.join(FP_DIR, uid)
    palm_path = os.path.join(PALM_DIR, uid)

    if not (os.path.isdir(face_path) and
            os.path.isdir(fp_path) and
            os.path.isdir(palm_path)):
        continue

    # -------- FILTER ONLY VALID IMAGE FILES ----------
    face_files = sorted([f for f in os.listdir(face_path)
                         if f.lower().endswith(('.jpg','.jpeg','.png'))])

    fp_files = sorted([f for f in os.listdir(fp_path)
                       if f.lower().endswith(('.jpg','.jpeg','.png'))])

    palm_files = sorted([f for f in os.listdir(palm_path)
                         if f.lower().endswith(('.jpg','.jpeg','.png'))])

    if len(face_files) == 0 or len(fp_files) == 0 or len(palm_files) == 0:
        continue

    min_count = min(len(face_files), len(fp_files), len(palm_files))

    for i in range(min_count):

        # ---------- FACE ----------
        face_file = os.path.join(face_path, face_files[i])
        img = cv2.imread(face_file)

        if img is None:
            print(f"Skipping unreadable face: {face_file}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(img)

        if not faces:
            continue

        face_emb = torch.tensor(faces[0].normed_embedding).float()

        # ---------- FINGERPRINT ----------
        fp_file = os.path.join(fp_path, fp_files[i])
        fp_img = cv2.imread(fp_file, 0)

        if fp_img is None:
            print(f"Skipping unreadable fingerprint: {fp_file}")
            continue

        fp_img = cv2.resize(fp_img, (128,128)) / 255.0
        fp_tensor = torch.tensor(fp_img).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            fp_emb = fp_model(fp_tensor)[0].cpu()

        # ---------- PALM ----------
        palm_file = os.path.join(palm_path, palm_files[i])
        palm_img = cv2.imread(palm_file, 0)

        if palm_img is None:
            print(f"Skipping unreadable palm: {palm_file}")
            continue

        palm_img = cv2.resize(palm_img, (128,128)) / 255.0
        palm_tensor = torch.tensor(palm_img).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            palm_emb = palm_model(palm_tensor)[0].cpu()

        # ---------- STORE ----------
        face_list.append(face_emb)
        fp_list.append(fp_emb)
        palm_list.append(palm_emb)
        label_list.append(label)

print("\nTotal valid samples created:", len(label_list))

if len(label_list) == 0:
    raise ValueError("No valid samples created. Check dataset paths.")

face_tensor = torch.stack(face_list)
fp_tensor = torch.stack(fp_list)
palm_tensor = torch.stack(palm_list)
labels = torch.tensor(label_list)

torch.save({
    "face": face_tensor,
    "fp": fp_tensor,
    "palm": palm_tensor,
    "labels": labels
}, os.path.join(ROOT,"transform","fusion_training_data.pt"))

print("fusion_training_data.pt successfully created.")