import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_image
import torch.nn as nn
import torch
from FCN import FCN32s
from deepv3 import DeepLabV3Plus
import numpy as np
from tqdm import tqdm
import functional as F
import random
import base

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data_root = {'train':'C:/Users/Ray/Desktop/DLCV/dlcv-fall-2023-hw1-rayshiue-main/hw1_data/p3_data/train/',\
             'valid':'C:/Users/Ray/Desktop/DLCV/dlcv-fall-2023-hw1-rayshiue-main/hw1_data/p3_data/validation/'}
save_root = 'C:/Users/Ray/Desktop/DLCV/dlcv-fall-2023-hw1-rayshiue-main/Problem3/checkpoints/DeepLabV3Plus/lr8-5_eph60_sigmoid_DiLOSS'

try: os.makedirs(save_root)
except: print('Save Root Exists')

colormap = [[0, 255, 255],[255, 255, 0],[255, 0, 255],[0, 255, 0],[0, 0, 255],[255, 255, 255],[0, 0, 0]]
color_masks = np.zeros([7,3,512,512])
for i in range(7):
    for j in range(512):
        for k in range(512):
            color_masks[i,:,j,k] =  colormap[i]

cm2lbl = np.zeros(256**3)
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i
def image2label(im):
    data = np.array(im, dtype='int32')
    idx = (data[0, :, :] * 256 + data[1, :, :]) * 256 + data[2, :, :]
    return np.array(cm2lbl[idx], dtype='int64')

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

h = transforms.RandomHorizontalFlip(p=1)
v = transforms.RandomHorizontalFlip(p=1)

class MyDataset(Dataset):
    def __init__(self, mask_paths, sat_paths, phase):
        self.mask_paths = mask_paths
        self.sat_paths = sat_paths
        self.phase = phase
        self.normalize = transforms.Compose([transforms.Normalize(mean=MEAN, std=STD)])
        self.resize = transforms.Compose([transforms.Resize(512, transforms.InterpolationMode.NEAREST)])
    def __len__(self):
        return len(self.mask_paths)
    
    def __getitem__(self, idx):
        sat, mask = read_image(self.sat_paths[idx]).float(), read_image(self.mask_paths[idx]).float()
        if self.phase == 'train':
            if random.random() < 0.5:
                sat, mask = h(sat), h(mask)
            if random.random() < 0.5:
                sat, mask = v(sat), v(mask)
            params = transforms.RandomResizedCrop.get_params(sat, scale=(0.8, 1.0), ratio=(1, 1))
            sat = self.resize(transforms.functional.crop(sat, *params))
            mask = self.resize(transforms.functional.crop(mask, *params))

        semantic_map = []
        for colour in color_masks:
            class_map = np.all(np.equal(mask.numpy(), colour), axis=0)
            semantic_map.append(class_map)

        return self.normalize(sat), np.stack(semantic_map, axis=0)      #image2label(mask)    # Used in model A

datasets, dataloaders= {}, {}
for phase in ['train', 'valid']:
    mask_paths, sat_paths = [], []
    f = True
    for filename in os.listdir(data_root[phase]):
        if f:
            mask_paths.append(os.path.join(data_root[phase], filename))
            f = False
        else:
            sat_paths.append(os.path.join(data_root[phase], filename))
            f = True
    datasets[phase] = MyDataset(mask_paths, sat_paths, phase)
    dataloaders[phase] = DataLoader(datasets[phase], batch_size=8, shuffle=(phase=='train'))


def mean_iou_score(pred, labels):
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
    return mean_iou

class DiceLoss(base.Loss):
    def __init__(self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        return 1 - F.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )
    

# criterion = nn.NLLLoss()        # Used in model A
criterion = DiceLoss()

# model = FCN32s().to(device)     # Used in model A
model = DeepLabV3Plus().to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=8e-5)

best_miou = 0
start = 0
epochs = 60
milestones = [40]
for epoch in range(start, epochs):
    print(f'\nEpoch: {epoch}')
    for phase in ['train', 'valid']:
        if phase == 'valid':
            model.eval()
        else:
            model.train()

        running_loss = 0.0
        mean_iou = 0.0
        corrects = 0
        all_preds = []
        all_labels = []

        bar = tqdm(total = len(dataloaders[phase]))
        for batch_idx, (sats, labels) in enumerate(dataloaders[phase]):
            bar.update(1) 
            sats, labels = sats.to(device), labels.to(device)
            if phase == 'valid':
                with torch.no_grad():
                    outputs = (model(sats) + h(model(h(sats))) + v(model(v(sats))) + v(h(model(v(h(sats))))))/4
                    outputs = nn.functional.log_softmax(outputs, dim=1)
                    loss = criterion(outputs, labels) 
            else:
                outputs = model(sats)
                outputs = nn.functional.log_softmax(outputs, dim=1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            preds = torch.Tensor.numpy(torch.argmax(outputs, axis=1).cpu())
            # labels = np.argmax(torch.Tensor.numpy(labels.cpu()), axis=1)
            labels = torch.Tensor.numpy(labels.cpu())
            corrects += np.sum(preds==labels)
            running_loss += loss.item()
            all_preds.append(preds)
            all_labels.append(labels)
        bar.close()

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        mean_iou = mean_iou_score(all_preds, all_labels)
        print(phase + f' loss: {running_loss/len(dataloaders[phase]):.4f}, mean_iou: {mean_iou:.4f}, accuracy: {corrects/len(datasets[phase])/512/512:.4f}')
        if phase == 'valid':
            torch.save(model.state_dict(), os.path.join(save_root, f'model_epoch{epoch}.pth'))
            if epoch in milestones:
                optimizer.param_groups[-1]['lr']*=0.1
            if mean_iou > best_miou:
                torch.save(model.state_dict(), os.path.join(save_root, f'model_best.pth'))
                best_miou = mean_iou
        with open(os.path.join(save_root,phase+'.txt'), 'a') as f:
            f.write(f'{running_loss/len(dataloaders[phase]):.4f} {mean_iou:.4f} {corrects/len(datasets[phase])/512/512:.4f}\n')
