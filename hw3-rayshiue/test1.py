import os
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import torch.nn as nn
import clip
from PIL import Image
import json
import sys

data_root, json_path, out_csv_path = sys.argv[1], sys.argv[2], sys.argv[3]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class MyDataset(Dataset):
    def __init__(self, img_names):
        self.img_names = img_names

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image = preprocess(Image.open(os.path.join(data_root,self.img_names[idx])))
        return image, self.img_names[idx]

img_names = []
for fname in os.listdir(data_root):
    img_names.append(fname)
dataloader = DataLoader(MyDataset(img_names), batch_size=8, shuffle=False)

with open(json_path) as f:
    data = json.load(f)

text = []
for key in data:
    text.append('This is a photograph of the ' + data[key] + '.')

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

corrects, total_num = 0, 0
softmax = nn.Softmax(dim=-1)
text = clip.tokenize(text).to(device)
text_features = model.encode_text(text)


f = open(out_csv_path, 'w')
f.write('filename,label\n')
for images, names in tqdm(dataloader):

    images = images.to(device)
    image_features = model.encode_image(images)
    
    logits_per_image, logits_per_text = model(images, text)
    probs = softmax(logits_per_image)
    preds = torch.argmax(probs, dim=1).cpu()

    for i in range(len(preds)):
        pred = int(preds[i])
        name = names[i]
        f.write(name+','+str(pred)+'\n')
f.close()
