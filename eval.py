
import torch
from torch.utils.data import DataLoader
from models.stead_model import STEAD
from dashcam_dataset import DashcamDataset
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = STEAD().to(device)
model.load_state_dict(torch.load('checkpoints/stead_driver.pt', map_location=device))
model.eval()

test_dataset = DashcamDataset('data/test')
test_loader = DataLoader(test_dataset, batch_size=1)

scores = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        recon, _ = model(xb)
        mse = ((recon - xb.flatten(1))**2).mean(dim=1)
        scores.append(mse.item())

print('Anomaly scores:', scores[:10])
