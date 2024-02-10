import os
from symbol import parameters
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import torch.nn as nn
import clip
from PIL import Image
import json
# from decoder import Decoder, Config
# from decoder_adapt import Decoder, Config
from decoder_lora import Decoder, Config
from tokenizer import BPETokenizer
import random
import numpy as np
import loralib as lora
import timm

data_root = '/D/ray/DLCV/HW3/hw3_data/p2_data/'
model_root = '/D/ray/DLCV/HW3/models/huge_lora/'

device = 'cuda:0'

try: os.makedirs(model_root)
except: print('model root exist')

maxlen = 60
json_path = '/D/ray/DLCV/HW3/hw3_data/p1_data/id2label.json'

class MyDataset(Dataset):
    def __init__(self, img_paths, all_inputs_tokens, all_gt_tokens, transforms):
        self.img_paths = img_paths
        self.all_inputs_tokens = all_inputs_tokens
        self.all_gt_tokens = all_gt_tokens
        self.transforms = transforms
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        image = self.transforms(Image.open(self.img_paths[idx]).convert('RGB'))
        choose = random.randint(0, len(self.all_gt_tokens[idx])-1)

        inputs_tokens = self.all_inputs_tokens[idx][choose]
        gt_tokens = self.all_gt_tokens[idx][choose]
        return image, torch.LongTensor(inputs_tokens), torch.LongTensor(gt_tokens)

dataloader = {}
encoding = BPETokenizer('./encoder.json', './vocab.bpe')

model = timm.create_model(
    # 'vit_base_patch16_224',
    # 'vit_large_patch16_224',
    'vit_huge_patch14_clip_336',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
).to(device)
model.eval()

data_config = timm.data.resolve_model_data_config(model)
# model, preprocess = clip.load("ViT-B/16", device=device)
# model, preprocess = clip.load("ViT-B/32", device=device)
# model, preprocess = clip.load("ViT-L/14", device=device)

model.eval()
for param in model.parameters():
    param.requires_grad = False

cfg = Config(checkpoint=os.path.join(data_root,'decoder_model.bin'))
decoder = Decoder(cfg)
lora.mark_only_lora_as_trainable(decoder)
decoder.to(device)
# decoder.load_state_dict(torch.load(model_root+'eph3.pth'))

for param in decoder.parameters():
    param.requires_grad = False
for param in decoder.enp.parameters():
    param.requires_grad = True
for j in range(12):
    for param in decoder.transformer.h[j].cross_attn.parameters():
        param.requires_grad = True
    for param in decoder.transformer.h[j].adapt.parameters():
        param.requires_grad = True
print(sum([p.numel() for p in decoder.parameters() if p.requires_grad]))


print('Encoding Captions...')
for phase in ['train', 'val']:
    
    transforms = timm.data.create_transform(**data_config, is_training=False)
    with open(os.path.join(data_root,phase+'.json')) as f:
        data = json.load(f)
    id2caption = {}
    for dt in data['annotations']:
        if dt['image_id'] in id2caption:
            id2caption[dt['image_id']].append(dt['caption'])
        else:
            id2caption[dt['image_id']] = [dt['caption']]
    img_paths, all_captions = [], []
    all_gt_tokens, all_inputs_tokens = [], []
    images_root = os.path.join(data_root,'images',phase)
    
    for dt in tqdm(data['images']):
        img_paths.append(os.path.join(images_root,dt['file_name']))
        inputs_tokens = []
        gt_tokens = []
        for s in id2caption[dt['id']]:
            tokens = encoding.encode(s)
            tokens.append(50256)
            gt_tokens.append(np.concatenate((tokens,[-100]*(maxlen-len(tokens)))))
            tokens.insert(0,50256)
            inputs_tokens.append(np.concatenate((tokens,[50256]*(maxlen-len(tokens)))))
        all_gt_tokens.append(gt_tokens)
        all_inputs_tokens.append(inputs_tokens)
    dataloader[phase] = DataLoader(MyDataset(img_paths, all_inputs_tokens, all_gt_tokens, transforms), \
                                   batch_size=32*(phase=='train')+24*(phase=='val'), shuffle=(phase=='train'))


lr=0.0001
optim = torch.optim.Adam(decoder.parameters())

print('Start Training...')
crit = nn.CrossEntropyLoss()
total_epochs = 10
for epoch in range(0,total_epochs):

    running_loss = {'train':0.0,'val':0.0}
    decoder.train()
    optim.param_groups[0]['lr']=lr*((total_epochs-epoch)/total_epochs)
    for phase in ['train', 'val']:
        if phase == 'val':
            decoder.eval()
        for images, input_tokens, gt_tokens in tqdm(dataloader[phase]):
            bsize = images.shape[0]
            gt_tokens = gt_tokens.to(device)
            image_features = model.forward_features(images.to(device))
            pred_outputs = decoder(input_tokens.to(device), image_features)
 
            if phase == 'val':
                with torch.no_grad():
                    loss = crit(pred_outputs.transpose(1,2), gt_tokens)
            else:
                loss = crit(pred_outputs.transpose(1,2), gt_tokens)
            if phase == 'train':
                loss.backward()
                optim.step()
                optim.zero_grad()
            running_loss[phase]+=loss.item()

    log = f'Epoch: {epoch},  Train Loss: {running_loss["train"]/len(dataloader["train"])},  Validation Loss: {running_loss["val"]/len(dataloader["val"])}\n'
    with open(model_root+'log.txt', 'a') as f:
        f.writelines(log)
    print(log)
    torch.save(decoder.state_dict(), model_root+f'eph{epoch}.pth')

