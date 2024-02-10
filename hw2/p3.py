import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from dann import Classifier, Extractor
import sys

data_root, out_root = sys.argv[1], sys.argv[2]

classifier_path = './DLCV_HW2_Models/svhn_classifier.pth'
encoder_path = './DLCV_HW2_Models/svhn_encoder.pth'


if 'svhn' in data_root:
    classifier_path = './DLCV_HW2_Models/svhn_classifier.pth'
    encoder_path = './DLCV_HW2_Models/svhn_encoder.pth'
elif 'usps' in data_root:
    classifier_path = './DLCV_HW2_Models/usps_classifier.pth'
    encoder_path = './DLCV_HW2_Models/usps_encoder.pth'

device = 'cuda:0'
batch_size = 32

class MyDataset(Dataset):
    def __init__(self, img_paths, img_names):
        self.img_paths = img_paths
        self.img_names = img_names

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        sample = read_image(self.img_paths[idx])/255
        if sample.shape[0] == 1:
            sample = sample.repeat(3, 1, 1)
        return sample, self.img_names[idx]
    
img_names, img_paths = [], []
for img_name in sorted(os.listdir(data_root)):
    img_names.append(img_name)
    img_paths.append(os.path.join(data_root,img_name))
    dataloaders = DataLoader(MyDataset(img_paths, img_names), batch_size=batch_size, shuffle=False)


classifier = Classifier().to(device)
encoder = Extractor().to(device)
classifier.load_state_dict(torch.load(classifier_path))
encoder.load_state_dict(torch.load(encoder_path))
encoder, classifier = encoder.eval(), classifier.eval()
    
out_file = open(out_root , 'w')
out_file.write('image_name,label\n')

for target_image, img_names in tqdm(dataloaders):
    target_image = target_image.to(device)
    tar_pred = torch.argmax(classifier(encoder(target_image)), dim=1)
    for i in range(len(tar_pred)):
        out_file.write(img_names[i] + ',' + str(int(tar_pred[i])) + '\n')
    
