import torch
from torch.utils.data import DataLoader
from models.stead_model import STEAD
from dataset_x3d import X3DFeatureDataset
from tqdm import tqdm
import os

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset setup (npy features)
data_root = "./X3D_Videos"
train_dataset = X3DFeatureDataset(root_dir=data_root, split="train")
test_dataset = X3DFeatureDataset(root_dir=data_root, split="test")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
# Make sure feature_dim matches your actual X3D features
feature_dim = 1024  # Adjust based on your .npy features shape [T, F]
seq_len = 16        # Number of frames/time steps per clip
model = STEAD(feature_dim=feature_dim, seq_len=seq_len).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for xb, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        xb = xb.to(device)  # [B, T, F]

        # Forward
        recon, _ = model(xb)

        # Target: temporal-averaged feature per clip
        target = xb.mean(dim=1)  # [B, F]
        loss = loss_fn(recon, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader):.6f}")

# Save model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/stead_driver.pt")
print("Training finished. Model saved to checkpoints/stead_driver.pt")