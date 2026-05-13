import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model_p import PalmprintNet
from dataset_p import PalmprintPairDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = PalmprintNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

def contrastive_loss(e1, e2, y, margin=1.0):
    d = F.pairwise_distance(e1, e2)
    return torch.mean(y*d**2 + (1-y)*torch.clamp(margin-d, min=0)**2)

ds = PalmprintPairDataset("../Data/palmprint_augmented")
dl = DataLoader(ds, batch_size=64, shuffle=True)

EPOCHS = 15
THRESH = 0.7

print("\n🚀 Training Palmprint Siamese CNN\n")
start = time.time()

for epoch in range(EPOCHS):
    loss_sum, correct, total = 0, 0, 0
    t0 = time.time()

    for x1, x2, y in dl:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        e1, e2 = model(x1), model(x2)
        loss = contrastive_loss(e1, e2, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        d = F.pairwise_distance(e1, e2)
        pred = (d < THRESH).float()
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss {loss_sum/len(dl):.4f} | "
          f"Acc {100*correct/total:.2f}% | "
          f"Time {time.time()-t0:.1f}s")

torch.save(model.state_dict(), "palm_model.pth")
print(f"\n💾 Saved palm_model.pth | Total time {(time.time()-start)/60:.2f} min")
