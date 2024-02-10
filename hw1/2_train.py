import torch
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

self_model_path = './self_checkpoints/lr3-4_eph40_bh160/model_best.pth'     # For setting B and D, self_model_path = pretrain_model_SL.pt
save_root = './train_checkpoints/C/lr-1_eph40_bh160/'
data_root = {'train':'C:/Users/Ray/Desktop/DLCV/dlcv-fall-2023-hw1-rayshiue-main/hw1_data/p2_data/office/train/',\
             'valid':'C:/Users/Ray/Desktop/DLCV/dlcv-fall-2023-hw1-rayshiue-main/hw1_data/p2_data/office/val'}

try: os.makedirs(save_root)
except: print('Save Root Exists')

model = models.resnet50(weights=None)
model.load_state_dict(torch.load(self_model_path))

for param in model.parameters():
    param.requires_grad = False     #For setting A, B and C, param.requires_grad = True
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True

model = torch.nn.Sequential(model, 
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm1d(num_features=1000),
                            torch.nn.Linear(in_features=1000, out_features=256, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm1d(num_features=256),
                            torch.nn.Linear(in_features=256, out_features=65, bias=True)
                            )
model.to(device)
torch.save(model.state_dict(), os.path.join(save_root, 'model_epoch0.pth'))

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
class MyDataset(Dataset):
    def __init__(self, image_paths, labels, phase):
        self.image_paths = image_paths
        self.labels = labels
        if phase == 'train':
            self.transforms = transforms.Compose([
            transforms.Resize(128, antialias=True),
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(128),
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
    dataloaders[phase] = DataLoader(datasets[phase], batch_size=128, shuffle=(phase=='train'))

lr = 0.1
start = 0
epochs = 60
milestones = [20, 40]
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001, nesterov=True)

best_acc = 0
tt = transforms.Compose([transforms.RandomCrop(128),])
for epoch in range(start, epochs):
    print(f'\nEpoch: {epoch}')
    for phase in ['train', 'valid']:
        if phase == 'valid':
            model.eval()
        else:
            model.train()
            
        running_loss = 0.0
        corrects = 0

        bar = tqdm(total = len(dataloaders[phase]))
        for batch_idx,(inputs, labels) in enumerate(dataloaders[phase]):
            bar.update(1)
            inputs, labels = inputs.to(device), labels.to(device)

            if phase == 'valid':
                with torch.no_grad():
                    for i in range(20):
                        input = tt(inputs).to(device)
                        if i==0:
                            outputs = model(input)
                        else:
                            outputs += model(input)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)/20
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            corrects += torch.sum(torch.argmax(outputs, dim=1)==labels)
            running_loss += loss.item()
        bar.close()
        
        loss, acc = running_loss/len(dataloaders[phase]), corrects/len(datasets[phase])
        print(phase + f'loss: {loss:.4f}, accuracy: {acc:.4f}')
        if phase=='valid':
            if epoch in milestones:
                optimizer.param_groups[-1]['lr']*=0.1
            if (epoch+1)%5==0:
                torch.save(model.state_dict(), os.path.join(save_root, f'model_epoch{epoch+1}.pth'))
            if acc > best_acc:
                torch.save(model.state_dict(), os.path.join(save_root, f'model_best.pth'))
                best_acc = acc
        with open(os.path.join(save_root,phase+'.txt'), 'a') as f:
            f.write(f'{loss:.4f} {acc:.4f}\n')
