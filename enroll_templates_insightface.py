import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

DATASET = "Data/augmented_face"
SAVE_PATH = "templates/face_db.npy"

os.makedirs("templates", exist_ok=True)

# ---------------- LOAD INSIGHTFACE ----------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

face_db = {}

print("📂 Creating templates using InsightFace...\n")

for person in sorted(os.listdir(DATASET)):
    person_dir = os.path.join(DATASET, person)

    # ✅ FIX: skip files like .DS_Store
    if not os.path.isdir(person_dir):
        continue

    embeddings = []

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            continue

        # use aligned ArcFace embedding
        embeddings.append(faces[0].normed_embedding)

    if len(embeddings) > 0:
        final_emb = np.mean(embeddings, axis=0)
        final_emb = final_emb / np.linalg.norm(final_emb)
        face_db[person] = final_emb
        print(f"✅ Enrolled: {person} ({len(embeddings)} images)")
    else:
        print(f"⚠️ Skipped: {person} (no valid faces)")

np.save(SAVE_PATH, face_db)
print("\n🎉 Templates created successfully!")
