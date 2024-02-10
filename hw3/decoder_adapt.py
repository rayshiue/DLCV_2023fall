import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import MultiheadAttention
class Config:

    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class CrossAttention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.q_attn = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.k_attn = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.v_attn = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x, enc_x):
        B, T, C = x.size() # batch, context, embedding
        B2, T2, C2 = enc_x.size()   #T2 = 197
        q = self.q_attn(x)
        k = self.k_attn(enc_x)
        v = self.v_attn(enc_x)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     #(batch, 60, 12, 64) => (batch, 12, 60, 64)
        k = k.view(B2, T2, self.n_head, C2 // self.n_head).transpose(1, 2)  #batch, 197, 12, 64 => (batch, 12, 197, 64)
        v = v.view(B2, T2, self.n_head, C2 // self.n_head).transpose(1, 2)  #batch, 197, 12, 64 => (batch, 12, 197, 64)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))     # (batch, 12, 60, 64) @ (batch, 12, 64, 197) = (batch, 12, 60, 197)
        # att = att.masked_fill(self.bias[:,:,:T,:T2] == 0, float('-inf'))    # (batch, 12, 60, 197)
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C)) # (batch, 12, 60, 197) @ (batch, 12, 197, 64) 
                                                                                 # = (batch, 12, 60, 64) => (batch, 60, 12, 64)

hidden_size = 256
class Adapter(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.down_project = nn.Linear(in_features=in_feature, out_features=hidden_size)
        self.up_project = nn.Linear(in_features=hidden_size, out_features=in_feature)

    def forward(self, x):
        input = x.clone()
        x = self.up_project(F.relu(self.down_project(x)))
        return x + input
    
class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.ln_3 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.cross_attn = CrossAttention(cfg)
        # self.cross_attn = MultiheadAttention(embed_dim = cfg.n_embd, num_heads = cfg.n_head)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))
        self.adapt = Adapter(cfg.n_embd)

    def forward(self, x, enc_x):

        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_3(x), enc_x) #add
        # x = x + self.cross_attn(self.ln_2(x), enc_x, enc_x)[0] #add
        x = x + self.mlp(self.ln_2(x))
        x  = self.adapt(x)
        return x, enc_x

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.enp = nn.Linear(1280, 768)
        # self.linear_align_img2embd = nn.Linear(512,cfg.n_embd)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            # h = nn.Sequential(*[Block(cfg) for _ in range(1)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x, enc_x):
        # print(x.shape)
        # enc_x = self.featureEmbd(enc_x)
        enc_x = self.enp(enc_x)
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))

        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)  
        for i in range(len(self.transformer.h)):
            x, enc_x = self.transformer.h[i](x, enc_x)

        x = self.lm_head(self.transformer.ln_f(x))
        return x 
