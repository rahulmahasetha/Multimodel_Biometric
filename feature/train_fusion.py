import torch
import numpy as np
from transformer_feature_fusion import TransformerFeatureFusion

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Dummy training data (replace with real embeddings if available)
N = 500
face_embs = torch.randn(N, 512).to(device)
fp_embs   = torch.randn(N, 256).to(device)
palm_embs = torch.randn(N, 256).to(device)
labels    = torch.randint(0, 50, (N,)).to(device)

model = TransformerFeatureFusion().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

model.train()

for epoch in range(10):
    optimizer.zero_grad()

    fused = model(face_embs, fp_embs, palm_embs)
    loss = criterion(fused, labels)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "fusion_transformer.pth")
print("✅ fusion_transformer.pth saved")
