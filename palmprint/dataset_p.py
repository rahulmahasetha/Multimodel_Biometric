import os
import cv2
import random
import torch
from torch.utils.data import Dataset

class PalmprintPairDataset(Dataset):
    def __init__(self, root, size=128):
        self.root = root
        self.size = size

        # ✅ Only valid identity folders
        self.people = [
            p for p in os.listdir(root)
            if os.path.isdir(os.path.join(root, p))
        ]

        if len(self.people) < 2:
            raise RuntimeError("❌ Need at least 2 identities")

    def _load(self, pid):
        person_dir = os.path.join(self.root, pid)

        imgs = [
            img for img in os.listdir(person_dir)
            if img.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # 🔁 Keep trying until a valid image is found
        while True:
            img_path = os.path.join(person_dir, random.choice(imgs))
            img = cv2.imread(img_path, 0)

            if img is None or img.size == 0:
                continue  # skip corrupted image

            try:
                img = cv2.resize(img, (self.size, self.size))
                img = img / 255.0
                return torch.tensor(img).unsqueeze(0).float()
            except:
                continue  # safety net

    def __len__(self):
        return 5000  # pairs per epoch

    def __getitem__(self, idx):
        # 50% genuine / 50% impostor
        if random.random() > 0.5:
            pid = random.choice(self.people)
            return self._load(pid), self._load(pid), torch.tensor(1.0)
        else:
            p1, p2 = random.sample(self.people, 2)
            return self._load(p1), self._load(p2), torch.tensor(0.0)
