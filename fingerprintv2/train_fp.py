import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model_fp import FingerprintNet
from dataset_fp import FingerprintPairDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = FingerprintNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def contrastive_loss(e1, e2, y, margin=1.0):
    d = F.pairwise_distance(e1, e2)
    return torch.mean(
        y * d.pow(2) + (1 - y) * torch.clamp(margin - d, min=0).pow(2)
    )

dataset = FingerprintPairDataset("../Data/fingerprint_augmented")
loader = DataLoader(dataset, batch_size=64, shuffle=True)

EPOCHS = 30
THRESHOLD = 0.7

print("\n🚀 Training Siamese ResNet-18 for Fingerprint Verification\n")
total_start = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    total_loss = 0
    correct = 0
    total = 0

    for x1, x2, y in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        e1 = model(x1)
        e2 = model(x2)

        loss = contrastive_loss(e1, e2, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        d = F.pairwise_distance(e1, e2)
        preds = (d < THRESHOLD).float()

        correct += (preds == y).sum().item()
        total += y.size(0)

    scheduler.step()

    acc = 100 * correct / total
    epoch_time = time.time() - epoch_start

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {total_loss/len(loader):.4f} | "
        f"Acc: {acc:.2f}% | "
        f"Time: {epoch_time:.1f}s"
    )

torch.save(model.state_dict(), "fp_model.pth")

print(f"\n💾 Model saved as fp_model.pth")
print(f"⏱️ Total training time: {(time.time()-total_start)/60:.2f} minutes")
