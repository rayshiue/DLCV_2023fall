import os
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import json
from decoder_adapt import Decoder, Config
# from decoder_lora import Decoder, Config
from tokenizer import BPETokenizer
import numpy as np
import timm
import sys

data_root, out_json_path, bin_path = sys.argv[1], sys.argv[2], sys.argv[3]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
decoder_path = './DLCV_HW3_Models/p2_decoder.pth'
maxlen = 60

class MyDataset(Dataset):
    def __init__(self, image_names, transforms):
        self.transforms = transforms
        self.image_names = image_names
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image = self.transforms(Image.open(os.path.join(data_root,self.image_names[idx])).convert('RGB'))
        return image, self.image_names[idx]

dataloader = {}
encoding = BPETokenizer('./encoder.json', './vocab.bpe')


model = timm.create_model(
    # 'vit_base_patch16_224',
    # 'vit_large_patch16_224',
    'vit_huge_patch14_clip_336',
    pretrained=True,
    num_classes=0,
).to(device)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

img_names = []
for fname in os.listdir(data_root):
    img_names.append(fname)
dataloader = DataLoader(MyDataset(img_names, transforms), batch_size=1, shuffle=False)


cfg = Config(checkpoint=bin_path)
decoder = Decoder(cfg).to(device)
my_state_dict = torch.load(decoder_path)
print('Number of Trainable Parameters: ', sum([p.numel() for m, p in my_state_dict.items()]))
decoder.load_state_dict(my_state_dict, strict=False)


encoding = BPETokenizer('./encoder.json', './vocab.bpe')
softmax = nn.Softmax(dim=-1)

for phase in ['val']:

    model.eval()
    decoder.eval()

    final_predictions = {}
    for images, img_names in tqdm(dataloader):

        bsize = images.shape[0]
        images = images.to(device)
        image_features = model.forward_features(images.to(device))

        generated_tokens = torch.tensor([[50256]]).expand(bsize,-1).to(device)
        for i in range(maxlen):
            outputs= decoder(generated_tokens.to(device), image_features)
            pred = torch.argmax(outputs, dim=-1)
            generated_tokens = torch.concat((generated_tokens,pred[:,-1][:,None]),dim=1)


        for i in range(bsize):
            tokens = generated_tokens[i][1:].tolist()
            try:
                predict_captions = tokens[:tokens.index(50256)]
            except:
                predict_captions = tokens
            final_predictions[img_names[i][:-4]] = encoding.decode(predict_captions)

with open(out_json_path, 'w') as f:
    json.dump(final_predictions, f)
