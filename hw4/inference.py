import torch
from collections import defaultdict
from rendering import *
from nerf import *
from dataset import KlevrDataset
from PIL import Image
import sys
import os
import numpy as np

data_root, out_root = sys.argv[1], sys.argv[2]
ckpt_path = './DLCV_HW4_Models/best.ckpt'

torch.backends.cudnn.benchmark = True
img_wh = (256, 256)

embedding_xyz = Embedding(3, 10)
embedding_dir = Embedding(3, 4)


def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)

nerf_coarse = NeRF()
nerf_fine = NeRF()

load_ckpt(nerf_coarse, ckpt_path, model_name='nerf_coarse')
load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')

nerf_coarse.cuda().eval()
nerf_fine.cuda().eval()

models = [nerf_coarse, nerf_fine]
embeddings = [embedding_xyz, embedding_dir]

N_samples = 64
N_importance = 256
use_disp = False
chunk = 1024*32*4

dataset = KlevrDataset(data_root, 'test')


@torch.no_grad()
def f(rays):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results

def normalize(x):
    return  (x*255).astype(np.uint8)

for idx, sample in enumerate(dataset):
    print(f'-----{idx}/{len(dataset)}', end='\r\r')

    rays = sample['rays'].cuda()
    results = f(rays)

    img_pred = normalize(results['rgb_fine'].view(img_wh[1], img_wh[0], 3).cpu().numpy())
    # depth_pred = visualize_depth(results['depth_fine'].view(img_wh[1], img_wh[0]))
    
    Image.fromarray(img_pred).save(os.path.join(out_root, f'{dataset.split_ids[idx]:05d}.png'))
    torch.cuda.synchronize()


