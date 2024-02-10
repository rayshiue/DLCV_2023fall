import torch
from byol_pytorch import BYOL
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import os
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
save_root = './self_checkpoints/lr3-4_eph40_bh160/'
self_data_root = "C:/Users/Ray/Desktop/DLCV/dlcv-fall-2023-hw1-rayshiue-main/hw1_data/p2_data/mini/train"

try: os.makedirs(save_root)
except: print('Save Root Exists')

model = models.resnet50(weights=None).to(device)
learner = BYOL(
    model,
    image_size = 128,
    hidden_layer = 'avgpool',
)

lr = 3e-4
optimizer = torch.optim.Adam(learner.parameters(), lr=lr)
milestones = [20]

class MyDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx]).float()
        return image
    
paths = []
for filename in os.listdir(self_data_root):
    paths.append(os.path.join(self_data_root, filename))
datasets = MyDataset(paths)
dataloaders = DataLoader(datasets, batch_size=128, shuffle=True)

epochs = 40
best_loss = 1e9
for epoch in range(epochs):
    print(f'\nEpoch: {epoch}')
    running_loss = 0.0

    bar = tqdm(total = len(dataloaders))
    for batch_idx, inputs in enumerate(dataloaders):
        bar.update(1)

        optimizer.zero_grad()
        loss = learner(inputs.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    bar.close()

    loss = running_loss/len(dataloaders)
    print(f'loss: {loss:.4f}')
    if loss < best_loss:
        torch.save(model.state_dict(), os.path.join(save_root, f'model_best.pth'))
    if (epoch+1)%5==0:
        torch.save(model.state_dict(), os.path.join(save_root, f'model_eph{epoch+1}.pth'))
    if epoch in milestones:
        optimizer.param_groups[-1]['lr']*=0.1
    with open(os.path.join(save_root,'loss.txt'), 'a') as f:
        f.write(f'{loss:.4f}\n')
