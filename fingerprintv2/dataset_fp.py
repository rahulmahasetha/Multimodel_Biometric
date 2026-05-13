import os
import cv2
import random
import torch
from torch.utils.data import Dataset

class FingerprintPairDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.people = sorted(os.listdir(root))

    def __len__(self):
        return 5000  # random pairs

    def _load_img(self, path):
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        return torch.tensor(img).unsqueeze(0).float()

    def __getitem__(self, idx):
        same = random.choice([0, 1])

        if same:
            p = random.choice(self.people)
            imgs = os.listdir(os.path.join(self.root, p))
            a, b = random.sample(imgs, 2)
            y = 1
        else:
            p1, p2 = random.sample(self.people, 2)
            a = random.choice(os.listdir(os.path.join(self.root, p1)))
            b = random.choice(os.listdir(os.path.join(self.root, p2)))
            y = 0

        img1 = self._load_img(os.path.join(self.root, p if same else p1, a))
        img2 = self._load_img(os.path.join(self.root, p if same else p2, b))

        return img1, img2, torch.tensor(y).float()
