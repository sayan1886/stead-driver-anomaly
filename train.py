
import torch
from torch.utils.data import DataLoader
from models.stead_model import STEAD
from dashcam_dataset import DashcamDataset
from tqdm import tqdm
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_dataset = DashcamDataset('data/train')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = STEAD().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(10):
    model.train()
    running = 0.0
    for xb, _ in tqdm(train_loader):
        xb = xb.to(device)
        recon, _ = model(xb)
        loss = loss_fn(recon, xb.flatten(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item()
    print(f'Epoch {epoch+1} loss: {running/len(train_loader):.6f}')

os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/stead_driver.pt')
