import torch
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

save_root = './FromScratch/EfficientNetB7/lr-1_eph200/'
data_root = {'train':'./hw1_data/p1_data/train_50/',\
             'valid':'./hw1_data/p1_data/val_50/'}

try: os.makedirs(save_root)
except: print('Save Root Exists')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = models.efficientnet_b7(weights=None)    # For model B, weights='IMAGENET1K_V1'
model.classifier = torch.nn.Linear(in_features=2560, out_features=50, bias=True)
model.to(device)
torch.save(model.state_dict(), os.path.join(save_root, 'model_epoch0.pth'))


MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
class MyDataset(Dataset):
    def __init__(self, image_paths, labels, phase):
        self.image_paths = image_paths
        self.labels = labels
        if phase == 'train':
            self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.7, 1.0), antialias=True), # For model B, size = 224
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        else:
            self.transforms = transforms.Compose([
                transforms.Normalize(mean=MEAN, std=STD)
        ])
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx]).float()
        return self.transforms(image), self.labels[idx]

datasets, dataloaders= {}, {}
for phase in ['train', 'valid']:
    paths, labels = [], []
    for filename in os.listdir(data_root[phase]):
        labels.append(int(filename.split('_')[0]))
        paths.append(os.path.join(data_root[phase], filename))
    datasets[phase] = MyDataset(paths, labels, phase)
    dataloaders[phase] = DataLoader(datasets[phase], batch_size=512, shuffle=(phase=='train')) # For model B, batch_size = 32


lr = 0.01
epochs = 80                # For model B, epochs = 15
milestones = [30, 60]      # For model B, milestones = [5, 10]

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001, nesterov=True)

best_acc = 0
for epoch in range(epochs):
    print(f'\nEpoch: {epoch}')
    for phase in ['train', 'valid']:
        if phase == 'valid':
            model.eval()
        else:
            model.train()
        running_loss = 0.0
        corrects = 0

        bar = tqdm(total = len(dataloaders[phase]))
        for inputs, labels in dataloaders[phase]:
            bar.update(1)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if phase == 'valid':
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            corrects += torch.sum(torch.argmax(outputs, dim=1)==labels)
            running_loss += loss.item()
        bar.close()
        
        loss, acc = running_loss/len(dataloaders[phase]), corrects/len(datasets[phase])
        print(phase + f'loss: {loss:.4f}, accuracy: {acc:.4f}')
        if phase=='valid':
            if epoch in milestones:
                optimizer.param_groups[-1]['lr']*=0.1
            if (epoch+1)%10==0:
                torch.save(model.state_dict(), os.path.join(save_root, f'model_epoch{epoch+1}.pth'))
            if acc > best_acc:
                torch.save(model.state_dict(), os.path.join(save_root, f'model_best.pth'))

        with open(os.path.join(save_root,phase+'.txt'), 'a') as f:
            f.write(f'{loss:.4f} {acc:.4f}\n')
