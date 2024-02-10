import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import torch.nn as nn

from net import MyUNet

data_root = '/D/ray/DLCV/HW2/hw2_data/digits/mnistm/data'
model_root = '/D/ray/DLCV/HW2/models/step700_SiLU_d160_block2conv_downcov/'
test_root = '/D/ray/DLCV/HW2/test_images/step700_SiLU_d160_block2conv_downcov/'
device = 'cuda:0'

try: os.makedirs(model_root)
except: print('model root exist')
try: os.makedirs(test_root)
except: print('test root exist')

data_csv = {'train':'/D/ray/DLCV/HW2/hw2_data/digits/mnistm/train.csv',\
            'valid':'/D/ray/DLCV/HW2/hw2_data/digits/mnistm/val.csv'}

class MyDataset(Dataset):
    def __init__(self, img_paths, labels):
        self.img_paths = img_paths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = read_image(self.img_paths[idx])/255
        return sample, self.labels[idx]

dataloaders = {}
for phase in ['train', 'valid']:
    with open(data_csv[phase], 'r') as f:
        img_paths, labels = [], []
        for s in f.readlines()[1:]:
            name, label = s.split(',')
            img_paths.append(os.path.join(data_root,name))
            labels.append(int(label))
        dataloaders[phase] = DataLoader(MyDataset(img_paths, labels), batch_size=128, shuffle=(phase=='train'))


def ddpm_schedules(beta1, beta2, T):

    beta_t = torch.arange(beta1, beta2+(beta2-beta1)/n_T, (beta2-beta1)/n_T,dtype=torch.float32)
    sqrt_beta_t = torch.sqrt(beta_t)#
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)#
    oneover_sqrta = 1 / torch.sqrt(alpha_t)#

    sqrtmab = torch.sqrt(1 - alphabar_t)#
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab#

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = self.sqrtab[_ts, None, None, None] * x + self.sqrtmab[_ts, None, None, None] * noise
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w = 0.0):

        x_i = torch.randn(n_sample, *size).to(device)
        c_i = torch.arange(0,10).to(device)
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free
        
        for i in tqdm(range(self.n_T, 0, -1)):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]

            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        return x_i
    

n_epoch = 20
milestones = [10]
n_T = 700
lr = 1e-4

ddpm = DDPM(nn_model=MyUNet(n_steps = n_T), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1).to(device)
start_eph = 0
optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

f = open(model_root+'loss.txt', 'a')
for ep in range(start_eph, n_epoch):
    print(f'epoch {ep}')
    for phase in ['train', 'valid']:
        if phase == 'train':
            ddpm.train()
        else:
            ddpm.eval()
            if ep in milestones:
                optim.param_groups[0]['lr']*=0.1

        running_loss = 0.0
        for x, c in tqdm(dataloaders[phase]):
            optim.zero_grad()
            x, c = x.to(device), c.to(device)
            loss = ddpm(x, c)
            if phase=='train':
                loss.backward()
            running_loss+=loss.item()
            optim.step()
        print(f'loss: {running_loss/len(dataloaders[phase])}')
        f.write(f'{running_loss/len(dataloaders[phase])} ')
    f.write('\n')
    torch.save(ddpm.state_dict(), model_root+f'eph{ep}.pth')
f.close()