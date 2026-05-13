import os
import cv2
import torch
import numpy as np
from model_p import PalmprintNet

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= LOAD MODEL =================
model = PalmprintNet().to(device)
model.load_state_dict(torch.load("palm_model.pth", map_location=device))
model.eval()

# ================= DATA =================
DATA = "../Data/palmprint_augmented"
db = {}

print("\n📂 Enrolling palmprint templates...\n")

for pid in sorted(os.listdir(DATA)):
    pdir = os.path.join(DATA, pid)

    # ✅ Skip non-directories (.DS_Store etc.)
    if not os.path.isdir(pdir):
        continue

    embs = []

    for img_name in os.listdir(pdir):
        img_path = os.path.join(pdir, img_name)

        # ✅ Only image files
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        im = cv2.imread(img_path, 0)

        # ✅ Skip unreadable images
        if im is None or im.size == 0:
            continue

        try:
            im = cv2.resize(im, (128, 128))
        except:
            continue

        im = im / 255.0
        im = torch.tensor(im).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            e = model(im).cpu().numpy()[0]

        embs.append(e)

    # ✅ Skip identity if no valid embeddings
    if len(embs) == 0:
        print(f"⚠️ Skipped {pid} (no valid images)")
        continue

    t = np.mean(embs, axis=0)
    t = t / np.linalg.norm(t)

    db[pid] = t
    print(f"✅ Enrolled {pid} ({len(embs)} images)")

# ================= SAVE =================
os.makedirs("templates", exist_ok=True)
np.save("palm_db.npy", db)

print("\n🎉 Palmprint templates saved successfully")
print(f"📊 Total enrolled identities: {len(db)}")
