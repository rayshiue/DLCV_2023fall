import torch
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import sys

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

images_csv, data_root, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

ids, names = [], []
with open(images_csv, 'r') as f:
    for s in f.readlines()[1:]:
        s = s.split(',')
        ids.append(s[0])
        names.append(s[1])

model = models.resnet50(weights=None)
model = torch.nn.Sequential(model, 
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm1d(num_features=1000),
                            torch.nn.Linear(in_features=1000, out_features=256, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm1d(num_features=256),
                            torch.nn.Linear(in_features=256, out_features=65, bias=True)
                            )

model.load_state_dict(torch.load('./DLCV_HW1_Models/hw1_2.pth'))
model.to(device).eval()

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
class MyDataset(Dataset):
    def __init__(self, ids, data_root, names):
        self.ids = ids
        self.data_root = data_root
        self.names = names
        self.transforms = transforms.Compose([
            transforms.Resize(128, antialias=True),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        image = read_image(os.path.join(self.data_root, self.names[idx])).float()
        return self.transforms(image), self.ids[idx], self.names[idx]

dataset = MyDataset(ids, data_root, names)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

out_file = open(output_path , 'w')
out_file.write('id,filename,label\n')

tt = transforms.Compose([transforms.RandomCrop(128),])
with torch.no_grad():
    corrects = 0
    bar = tqdm(total = len(dataloader))
    for batch_idx,(inputs, ids, names) in enumerate(dataloader):
        bar.update(1)
        for i in range(20):
            input = tt(inputs).to(device)
            if i==0:
                outputs = model(input)
            else:
                outputs += model(input)
        preds = torch.argmax(outputs, dim=1)
        out_file.write(ids[0] + ',' + names[0] + ',' + str(int(preds[0])) + '\n')
    bar.close()
    out_file.close()
