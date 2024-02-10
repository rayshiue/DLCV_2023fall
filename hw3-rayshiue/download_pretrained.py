import torch
import timm
import clip

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = timm.create_model(
    'vit_huge_patch14_clip_336',
    pretrained=True,
    num_classes=0,
)
model, preprocess = clip.load("ViT-B/32", device=device)