import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from model_p import PalmprintNet   # change model if needed
from verify_p import verify_palm   # change verify fn if needed

# ================= CONFIG =================
DATASET = "../Data/palmprint_augmented"
TEMPLATE_DB = "palm_db.npy"
IMAGE_SIZE = 128
THRESHOLDS = np.linspace(0.3, 0.95, 100)

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= LOAD MODEL =================
model = PalmprintNet().to(device)
model.load_state_dict(torch.load("palm_model.pth", map_location=device))
model.eval()

# ================= LOAD TEMPLATES =================
db = np.load(TEMPLATE_DB, allow_pickle=True).item()

# ================= LOAD ALL EMBEDDINGS =================
print("\n📂 Extracting embeddings...")
samples = []

for pid in sorted(os.listdir(DATASET)):
    pdir = os.path.join(DATASET, pid)
    if not os.path.isdir(pdir):
        continue

    for img_name in os.listdir(pdir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(pdir, img_name)
        img = cv2.imread(img_path, 0)
        if img is None:
            continue

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            emb = model(img).cpu().numpy()[0]

        emb = emb / np.linalg.norm(emb)
        samples.append((pid, emb))

print(f"✅ Total samples: {len(samples)}")

# ================= EVALUATION =================
print("\n📊 Evaluating verification performance...\n")

best_acc = 0
best_th = 0
best_far = 0
best_frr = 0

total_comparisons = 0

for th in THRESHOLDS:
    tp = fp = tn = fn = 0

    for i in range(len(samples)):
        pid1, emb1 = samples[i]

        for pid2, emb2 in samples:
            score = np.dot(emb1, emb2)
            same = pid1 == pid2
            match = score >= th

            if match and same:
                tp += 1
            elif match and not same:
                fp += 1
            elif not match and not same:
                tn += 1
            elif not match and same:
                fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total
    far = fp / (fp + tn + 1e-6)
    frr = fn / (fn + tp + 1e-6)

    if acc > best_acc:
        best_acc = acc
        best_th = th
        best_far = far
        best_frr = frr
        total_comparisons = total

# ================= REPORT =================
print("📊 VERIFICATION ACCURACY REPORT")
print("=================================")
print(f"Total Comparisons : {total_comparisons}")
print(f"Best Threshold    : {best_th:.2f}")
print(f"Accuracy          : {best_acc*100:.2f}%")
print(f"FAR               : {best_far*100:.2f}%")
print(f"FRR               : {best_frr*100:.2f}%")
print("=================================")
