import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from dann import Classifier, Extractor, Discriminator


data_root = {'mnistm':'C:/Users/Ray/Desktop/DLCV/hw2-rayshiue-main/hw2_data/hw2_data/digits/mnistm/',\
             'svhn':'C:/Users/Ray/Desktop/DLCV/hw2-rayshiue-main/hw2_data/hw2_data/digits/svhn/',\
             'usps':'C:/Users/Ray/Desktop/DLCV/hw2-rayshiue-main/hw2_data/hw2_data/digits/usps/'}

data_csv = {'train':'C:/Users/Ray/Desktop/DLCV/hw2-rayshiue-main/hw2_data/hw2_data/digits/mnistm/train.csv',\
            'valid':'C:/Users/Ray/Desktop/DLCV/hw2-rayshiue-main/hw2_data/hw2_data/digits/mnistm/val.csv'}

save_root = 'C:/Users/Ray/Desktop/DLCV/hw2-rayshiue-main/P3/mnist2svhn/'
# save_root = 'C:/Users/Ray/Desktop/DLCV/hw2-rayshiue-main/P3/mnist2usps/'
try:os.makedirs(save_root)
except: None
device = 'cuda:0'
batch_size = 64

class MyDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, img_paths, labels, phase, dataname):
        self.dataname = dataname
        self.img_paths = img_paths
        self.labels = labels
        self.images = []
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #sample = self.transform(self.images[idx])
        sample = read_image(self.img_paths[idx])/255
        if self.dataname == 'usps':
            sample = sample.repeat(3, 1, 1)
        return sample, self.labels[idx]
    
# mnistm, svhn, usps = {}, {}, {}
dataloaders = {'mnistm':{},'svhn':{},'usps':{}}
for dataname in ['mnistm','svhn','usps']:
    root = data_root[dataname]
    for phase in ['train', 'val']:
        with open(root+phase+'.csv', 'r') as f:
            img_paths, labels = [], []
            for s in f.readlines()[1:]:
                name, label = s.split(',')
                img_paths.append(os.path.join(root,'data',name))
                labels.append(int(label))
            dataloaders[dataname][phase] = DataLoader(MyDataset(img_paths, labels, phase, dataname), batch_size=batch_size, shuffle=(phase=='train'))


classifier = Classifier().to(device)
encoder = Extractor().to(device)
discriminator = Discriminator().to(device)

lr = 0.01
epochs = 40               
milestones = [20]   

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    list(encoder.parameters()) +
    list(classifier.parameters()) +
    list(discriminator.parameters()),
    lr=lr,
    momentum=0.9)

best_acc = 0
    
classifier_criterion = nn.CrossEntropyLoss().cuda()
discriminator_criterion = nn.CrossEntropyLoss().cuda()
best_acc = 0
for epoch in range(epochs):
    print(f'\nEpoch: {epoch}')
    # for phase in ['train', 'val']:
    for phase in ['train', 'val']:
        if phase == 'val':
            encoder, classifier, discriminator = encoder.eval(), classifier.eval(), discriminator.eval()
     

        loss, corrects, num = 0.0, 0.0, 0

        source_train_loader, target_train_loader = dataloaders['mnistm'][phase], dataloaders['svhn'][phase]
        # source_train_loader, target_train_loader = dataloaders['mnistm'][phase], dataloaders['usps'][phase]
        start_steps = epoch * len(source_train_loader)
        total_steps = epochs * len(target_train_loader)


        bar = tqdm(total = min(len(source_train_loader),len(target_train_loader)))
        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
            bar.update(1)
            optimizer.zero_grad()

            source_image, source_label = source_data
            num += source_image.shape[0]
            # print(source_image.shape)
            target_image, target_label = target_data
            
            source_image, source_label = source_image.cuda(), source_label.cuda()
            target_image, target_label = target_image.cuda(), target_label.cuda()
            combined_image = torch.cat((source_image, target_image), 0)

            p = float(batch_idx + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            combined_feature = encoder(combined_image)
            source_feature = encoder(source_image)

            # 1.Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)
            # print(class_pred.shape, source_label.shape)
            # 2. Domain loss
            domain_pred = discriminator(combined_feature, alpha)

            domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
            domain_loss = discriminator_criterion(domain_pred, domain_combined_label)
            
            # print(domain_pred.shape, domain_combined_label.shape)
            total_loss = class_loss + domain_loss
            if phase == 'train':
                total_loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 50 == 0:
                print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                batch_idx * len(target_image), len(target_train_loader.dataset), 100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(), domain_loss.item()))
            tar_pred = classifier(encoder(target_image))
            corrects += torch.sum(torch.argmax(tar_pred, dim=1)==target_label)
        bar.close()
        acc = corrects/num
        print(f'\n\tLoss: {total_loss.item():.6f}\tClass Loss: {class_loss.item():.6f}\tDomain Loss: {domain_loss.item():.6f}\taccuracy: {acc:.4f}')
            # loss += loss.item()/len(dataloaders['mnistm'][phase])
        
        # acc = corrects/num
        # loss, acc = running_loss/len(dataloaders[phase])
        # print(phase + f'loss: {loss:.4f}, accuracy: {acc:.4f}')
        if phase=='val':
            if epoch in milestones:
                optimizer.param_groups[-1]['lr']*=0.1
            rt = save_root+ f'epoch{epoch+1}'
            try: os.makedirs(rt)
            except: a = 1
            torch.save(encoder.state_dict(), os.path.join(rt, f'encoder.pth'))
            torch.save(discriminator.state_dict(), os.path.join(rt, f'discriminator.pth'))
            torch.save(classifier.state_dict(), os.path.join(rt, f'classifier.pth'))

            # if (epoch+1)%10==0:
            #     
            if acc > best_acc:
                rt = save_root+ f'best'
                try: os.makedirs(rt)
                except: a = 1
                torch.save(encoder.state_dict(), os.path.join(rt, f'encoder.pth'))
                torch.save(discriminator.state_dict(), os.path.join(rt, f'discriminator.pth'))
                torch.save(classifier.state_dict(), os.path.join(rt, f'classifier.pth'))
                best_acc = acc
            #     torch.save(model.state_dict(), os.path.join(save_root, f'model_best.pth'))
            #     best_acc = acc
        # with open(os.path.join(save_root,phase+'.txt'), 'a') as f:
        #     f.write(f'{loss:.4f} {acc:.4f}\n')

