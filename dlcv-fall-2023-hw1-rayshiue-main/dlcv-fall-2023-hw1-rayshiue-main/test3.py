import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_image
import torch
import numpy as np
from tqdm import tqdm
from deepv3 import DeepLabV3Plus
import sys
import PIL

device = 'cuda:0'
save_root = './DLCV_HW1_Models/hw1_3.pth'

data_root, output_path = sys.argv[1], sys.argv[2]

class MyDataset(Dataset):
    def __init__(self, sat_paths, names):
        self.sat_paths = sat_paths
        self.names = names
        self.normalize = transforms.Compose([
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        sat = read_image(self.sat_paths[idx]).float()
        return self.normalize(sat), self.names[idx]

sat_paths, names = [], []
for filename in sorted(os.listdir(data_root)):
    sat_paths.append(os.path.join(data_root, filename))
    names.append(filename[:-4])

datasets = MyDataset(sat_paths, names)
dataloaders = DataLoader(datasets, batch_size=4, shuffle=False)

model = DeepLabV3Plus()
model.load_state_dict(torch.load(save_root))
model.to(device).eval()
with torch.no_grad():
    colormap = np.array([[0, 255, 255],[255, 255, 0],[255, 0, 255],[0, 255, 0],[0, 0, 255],[255, 255, 255],[0, 0, 0]]).astype(np.uint8)
    bar = tqdm(total = len(dataloaders))
    for sats, names in dataloaders:
        bar.update(1) 
        outputs = model(sats.to(device))
        preds = torch.Tensor.numpy(torch.argmax(outputs, axis=1).cpu())
        for i in range(len(preds)):
            outimg = colormap[preds[i]]
            im = PIL.Image.fromarray(outimg)
            im.save(os.path.join(output_path, names[i]+".png"))
    bar.close()

            