import os
import cv2
import torch
import numpy as np
from model_fp import FingerprintNet

# ================= CONFIG =================
DATASET_DIR = "../Data/fingerprint_augmented"
MODEL_PATH = "fp_model.pth"
SAVE_PATH = "fp_db.npy"
IMAGE_SIZE = 128

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= LOAD MODEL =================
model = FingerprintNet().to(device)

if not os.path.exists(MODEL_PATH):
    print("❌ fp_model.pth not found. Train the model first.")
    exit()

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Loaded trained fingerprint model")

# ================= CLAHE (Fingerprint Enhancement) =================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ================= ENROLL TEMPLATES =================
fp_db = {}

print("\n📂 Enrolling fingerprint templates...\n")

for person in sorted(os.listdir(DATASET_DIR)):
    person_dir = os.path.join(DATASET_DIR, person)

    # Skip files like .DS_Store
    if not os.path.isdir(person_dir):
        continue

    embeddings = []

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        img = cv2.imread(img_path, 0)
        if img is None:
            continue

        # ---------- PREPROCESS ----------
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = clahe.apply(img)          # ✅ IMPORTANT CHANGE
        img = img / 255.0

        img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)

        # ---------- EMBEDDING ----------
        with torch.no_grad():
            emb = model(img).cpu().numpy()[0]

        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)

    if len(embeddings) == 0:
        print(f"⚠️ Skipped {person} (no valid images)")
        continue

    # ---------- TEMPLATE AGGREGATION ----------
    final_emb = np.mean(embeddings, axis=0)
    final_emb = final_emb / np.linalg.norm(final_emb)

    fp_db[person] = final_emb
    print(f"✅ Enrolled {person} ({len(embeddings)} images)")

# ================= SAVE DATABASE =================
os.makedirs("templates", exist_ok=True)
np.save("fp_db.npy", fp_db)

print("\n🎉 Fingerprint templates created successfully!")
print(f"👥 Total identities enrolled: {len(fp_db)}")
print("📁 Saved as fp_db.npy")
