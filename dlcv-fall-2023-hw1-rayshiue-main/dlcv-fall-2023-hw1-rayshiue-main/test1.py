import torch
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import sys

data_root, output_path = sys.argv[1], sys.argv[2]

model_path = './DLCV_HW1_Models/hw1_1.pth'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = models.efficientnet_b7(weights=None)
model.classifier = torch.nn.Linear(in_features=2560, out_features=50, bias=True)
model.load_state_dict(torch.load(model_path))
model.to(device).eval()

MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
class MyDataset(Dataset):
    def __init__(self, image_paths, names):
        self.image_paths = image_paths
        self.names = names
        self.transforms = transforms.Compose([
                transforms.Resize(224, antialias=True),
                transforms.Normalize(mean=MEAN, std=STD)
        ])
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx]).float()
        return self.transforms(image), self.names[idx]

paths, names = [], []
for filename in sorted(os.listdir(data_root)):
    if filename[-4:]=='.png':
        names.append(filename)
        paths.append(os.path.join(data_root, filename))

dataset = MyDataset(paths, names)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

out_file = open(output_path , 'w')
out_file.write('filename,label\n')

bar = tqdm(total = len(dataloader))
with torch.no_grad():
    for inputs, names in dataloader:
        bar.update(1)
        outputs = model(inputs.to(device))
        preds = torch.argmax(outputs, dim=1)
        for i in range(len(preds)):
            out_file.write(names[i] + ',' + str(int(preds[i])) + '\n')
bar.close()
out_file.close()
