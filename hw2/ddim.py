import os
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import torch.nn as nn
from UNet import UNet
import sys

noise_root, out_root, model_path = sys.argv[1], sys.argv[2], sys.argv[3]
# noise_root = '/D/ray/DLCV/HW2/hw2_data/face/noise/'       
# out_root = './test'
# model_path = '/D/ray/DLCV/HW2/hw2_data/face/UNet.pt'
# data_root = '/D/ray/DLCV/HW2/hw2_data/digits/mnistm/data'
device = 'cuda:0'

def ddpm_schedules():

    n_timestep=1000
    linear_start=1e-4
    linear_end=2e-2
    beta_t = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    sqrta = torch.sqrt(alpha_t)
    oneover_sqrta = 1 / sqrta

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

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
    def __init__(self, nn_model, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules().items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))
    def sigma(self, eta, alpha, pre_alpha):
        return eta*torch.sqrt((1-pre_alpha)/(1-alpha))*torch.sqrt(1-alpha/pre_alpha)
    def spherical_linear_interpolation(self, alpha, x1, x2):
        theta = torch.acos(torch.matmul(torch.t(x1), x2) / torch.norm(x1) / torch.norm(x2))
        x = torch.sin((1-alpha)*theta)/torch.sin(theta)*x1 \
          + torch.sin(alpha*theta)/torch.sin(theta)*x2
        return x
    def sample(self, n_sample, size, device):

        
        # new_x = torch.zeros(3,256,256)
        # for i in range(3):
            # new_x[i] = self.spherical_linear_interpolation(0.5,x_i[0,i],x_i[1,i])
        # xxx = self.spherical_linear_interpolation(0.5,x_i[0],x_i[1])
        # x_i[0] = new_x
        # all_x = [] 
        
        # for eta in [0,0.25,0.5,0.75,1]:
        # for alpha in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        x_i = []
        for j in range(10):
            x_i.append(torch.load(os.path.join(noise_root,f'0{j}.pt')))
        x_i = torch.stack(x_i).squeeze()
        # inter_x = 
        # new_x = torch.zeros(1,3,256,256).to(device)
        # for i in range(3):
            # new_x[0,i] = self.spherical_linear_interpolation(alpha,x_i[0,i],x_i[1,i])
            # new_x[0,i] = (1-alpha)*x_i[0,i] + alpha*x_i[1,i]
        # x_i = new_x
        time_steps = 50
        step_size = int(n_T/time_steps)
        for i in tqdm(range(n_T-1, 1, -step_size)):
            
            t_is = torch.tensor([i+1]).to(device)
            t_is = t_is.repeat(n_sample)

            # double batch
            # x_i = x_i.repeat(2,1,1,1)
            # t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, t_is)

            if i-step_size < 0:
                pi = 0
            else:
                pi = i-step_size

            eta = 0
            sgm = self.sigma(eta, self.alphabar_t[i], self.alphabar_t[pi])
            
            x_0 = (x_i - eps * torch.sqrt(1.0-self.alphabar_t[i]))/ torch.sqrt(self.alphabar_t[i])
            x_i = torch.sqrt(self.alphabar_t[pi]) * x_0 + torch.sqrt(1.0-self.alphabar_t[pi]-sgm**2) * eps + sgm*z

        # if alpha == 0:
        #     all_x = x_i
        # else:
        #     all_x = torch.cat((all_x,x_i))
        # for i in range(len(all_x)):
        #     vmin, vmax = torch.min(all_x[i]), torch.max(all_x[i])
        #     all_x[i] = ((all_x[i] - vmin)/(vmax - vmin))
        # all_x = make_grid(all_x,nrow=11)
        # save_image(all_x,'linear.png')
        # print(all_x.shape)
        # x_i_store = np.array(x_i_store)
        return x_i
    
def beta_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    return betas


n_T = 1000

net = UNet()
net.load_state_dict(torch.load(model_path))
ddpm = DDPM(nn_model=net, n_T=n_T, device=device, drop_prob=0.1)
ddpm.to(device)
ddpm.eval()

with torch.no_grad():
    n_sample = 10
    x_gen = ddpm.sample(n_sample, (3, 256, 256), device)
    for i in range(0,n_sample):
        vmin, vmax = torch.min(x_gen[i]), torch.max(x_gen[i])
        x_gen[i] = ((x_gen[i] - vmin)/(vmax - vmin))
        save_image(x_gen[i],os.path.join(out_root,f'{int(i):02d}.png'))
