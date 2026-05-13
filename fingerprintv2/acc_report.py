import os
import cv2
import numpy as np
import torch
from metrics import cosine_similarity
from model_fp import FingerprintNet  # change model for face/palm

# ================= CONFIG =================
DATASET = "../Data/fingerprint_augmented"
DB_PATH = "fp_db.npy"
MODEL_PATH = "fp_model.pth"
IMAGE_SIZE = 128

THRESHOLDS = np.arange(0.2, 0.9, 0.05)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= LOAD MODEL =================
model = FingerprintNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ================= LOAD DB =================
db = np.load(DB_PATH, allow_pickle=True).item()

scores = []
labels = []

# ================= EVALUATION =================
for person in os.listdir(DATASET):
    person_dir = os.path.join(DATASET, person)
    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        img = cv2.imread(img_path, 0)
        if img is None:
            continue

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255.0
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            emb = model(img).cpu().numpy()[0]

        emb = emb / np.linalg.norm(emb)

        for pid, template in db.items():
            score = cosine_similarity(emb, template)
            scores.append(score)
            labels.append(1 if pid == person else 0)

# ================= METRICS =================
best_acc = 0
best_thr = 0
best_far = 0
best_frr = 0

for t in THRESHOLDS:
    preds = [1 if s >= t else 0 for s in scores]

    correct = sum(p == y for p, y in zip(preds, labels))
    acc = correct / len(labels)

    FAR = sum(p == 1 and y == 0 for p, y in zip(preds, labels)) / max(1, sum(l == 0 for l in labels))
    FRR = sum(p == 0 and y == 1 for p, y in zip(preds, labels)) / max(1, sum(l == 1 for l in labels))

    if acc > best_acc:
        best_acc = acc
        best_thr = t
        best_far = FAR
        best_frr = FRR

# ================= REPORT =================
print("\n📊 VERIFICATION ACCURACY REPORT")
print("=================================")
print(f"Total Comparisons : {len(scores)}")
print(f"Best Threshold    : {best_thr:.2f}")
print(f"Accuracy          : {best_acc*100:.2f}%")
print(f"FAR               : {best_far*100:.2f}%")
print(f"FRR               : {best_frr*100:.2f}%")
print("=================================")
