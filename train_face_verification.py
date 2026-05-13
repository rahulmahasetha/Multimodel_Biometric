import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FaceDataset
from model import FaceNet
from arcface import ArcFace
import os

# ================= CONFIG =================
DATA = "Data/augmented_face"
EPOCHS = 25
BATCH = 32
LR = 1e-4

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= DATA =================
dataset = FaceDataset(DATA)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

# ================= MODEL =================
model = FaceNet(512).to(device)
arcface = ArcFace(512, num_classes=len(dataset.labels)).to(device)

optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(arcface.parameters()),
    lr=LR
)

criterion = nn.CrossEntropyLoss()

os.makedirs("checkpoints", exist_ok=True)

# ================= TRAIN =================
for epoch in range(EPOCHS):
    model.train()
    arcface.train()

    epoch_loss = 0
    correct = 0
    total = 0

    start_time = time.time()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        embeddings = model(images)
        logits = arcface(embeddings, labels)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # -------- ACCURACY (SANITY ONLY) --------
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        epoch_loss += loss.item()

    # -------- STATS --------
    epoch_time = time.time() - start_time
    acc = 100.0 * correct / total
    avg_loss = epoch_loss / len(loader)
    imgs_per_sec = total / epoch_time

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {avg_loss:.4f} | "
        f"Train Acc: {acc:.2f}% | "
        f"Time: {epoch_time:.1f}s | "
        f"Speed: {imgs_per_sec:.1f} img/s"
    )

    torch.save(model.state_dict(), "checkpoints/face_model.pth")

print("✅ Training complete")

 