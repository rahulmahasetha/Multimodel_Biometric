import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ================= PATH =================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from transform.transformer_feature_fusion import TransformerFeatureFusion

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= CONFIG =================
EMBED_DIM = 512
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-4

# ================= LOAD DATA =================
data_path = os.path.join(ROOT, "transform", "fusion_training_data.pt")

if not os.path.exists(data_path):
    raise FileNotFoundError("fusion_training_data.pt not found!")

data = torch.load(data_path)

face_feats = data["face"]
fp_feats   = data["fp"]
palm_feats = data["palm"]
labels     = data["labels"]

print("Total samples:", len(labels))

# ================= REMAP LABELS =================
unique_labels = torch.unique(labels)
label_map = {old.item(): idx for idx, old in enumerate(unique_labels)}
new_labels = torch.tensor([label_map[l.item()] for l in labels])

NUM_CLASSES = len(unique_labels)
print("Total identities:", NUM_CLASSES)

# ================= MOVE TO DEVICE =================
face_feats = face_feats.to(device)
fp_feats   = fp_feats.to(device)
palm_feats = palm_feats.to(device)
new_labels = new_labels.to(device)

# ================= DATA LOADER =================
dataset = TensorDataset(face_feats, fp_feats, palm_feats, new_labels)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ================= MODEL =================
fusion_model = TransformerFeatureFusion(embed_dim=EMBED_DIM).to(device)
classifier = nn.Linear(EMBED_DIM, NUM_CLASSES).to(device)

optimizer = optim.Adam(
    list(fusion_model.parameters()) + list(classifier.parameters()),
    lr=LR
)

criterion = nn.CrossEntropyLoss()

print("\nTraining fusion model...\n")

# ================= TRAIN LOOP =================
for epoch in range(EPOCHS):

    total_loss = 0

    for face, fp, palm, label in loader:

        optimizer.zero_grad()

        fused = fusion_model(face, fp, palm)
        logits = classifier(fused)

        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

# ================= SAVE MODEL =================
save_path = os.path.join(ROOT, "transform", "fusion_transformer.pth")

torch.save(fusion_model.state_dict(), save_path)

print("\nFusion model saved successfully at:")
print(save_path)