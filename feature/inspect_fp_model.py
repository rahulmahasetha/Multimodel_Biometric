import torch

state = torch.load("../fingerprint/fp_model.pth", map_location="cpu")

print("\n=== Fingerprint model state_dict keys ===")
for k in state.keys():
    print(k)
