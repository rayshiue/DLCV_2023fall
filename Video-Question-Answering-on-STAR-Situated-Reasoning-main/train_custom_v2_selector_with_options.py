import os
import sys
import argparse
import time
import datetime
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path

import math

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear
from torch.utils.tensorboard import SummaryWriter

import timm.optim.optim_factory as optim_factory

from collections import defaultdict, deque
import copy
from typing import Optional, Tuple, List, Iterable
from dataclasses import dataclass
from sentencepiece import SentencePieceProcessor

import clip_selector

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
'''logging level: NOSET, DEBUG, INFO, WARNING, ERROR, CRITICAL'''
from logging import getLogger
logger = getLogger()

# get args ============================================================================================================
def get_args_parser():
    parser = argparse.ArgumentParser('STAR training with inference', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * 1(# gpus)')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_model_path', default='./pretrained/llama/', type=str, help='path of llama model')
    parser.add_argument('--model', default='7B', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--adapter_layer', type=int, default=32, metavar='LENGTH', help='the number of adapter layer')
    parser.add_argument('--adapter_len', type=int, default=10, metavar='LENGTH', help='the adapter length')
    parser.add_argument('--max_seq_len', type=int, default=150, metavar='LENGTH', help='the maximum sequence length')
    parser.add_argument('--max_feats', type=int, default=30, metavar='LENGTH', help='the maximum feature length')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)') # 0.16
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)') # 0.0028125
    parser.add_argument('--blr', type=float, default=0.36, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256') # 1e-3
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--constant_lr', action='store_true', help='do not decay learning rate')

    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    parser.add_argument('--bias', type=float, default=3., help='attention bias')
    parser.add_argument('--tau', type=float, default=100., help='tau')
    parser.add_argument('--sub', action='store_true', help='subtitles for VLEP and TVQA')
    
    parser.add_argument('--split', type=str, default='train', help='split')

    return parser.parse_args()
# selector ============================================================================================================
class Selector(nn.Module):
    def __init__(self, topk, selection_method='gumbel', q_dim=768, dim=768):
        super(Selector, self).__init__()
        self.linear_Q = nn.Linear(q_dim, dim)
        self.norm_Q = nn.LayerNorm(dim, eps=1e-12)

        self.linear_K = nn.Linear(dim, dim)
        self.norm_K = nn.LayerNorm(dim, eps=1e-12)

        self.topk = topk
        self.selection_method = selection_method

    # @staticmethod
    # def sample_gumbel(n, k):
    #     unif = torch.distributions.Uniform(0, 1).sample((n, k))
    #     g = -torch.log(-torch.log(unif))
    #     return g

    # # @staticmethod
    # def sample_gumbel_softmax(self, pi, temperature):
    #     n, k = pi.shape
    #     # dbg.set_trace()
    #     g = self.sample_gumbel(n, k).to(pi.device)
    #     h = (g + torch.log(pi)) / temperature
    #     h_max = h.max(dim=1, keepdim=True)[0]
    #     h = h - h_max
    #     cache = torch.exp(h)
    #     #     print(pi, torch.log(pi), intmdt)
    #     y = cache / cache.sum(dim=-1, keepdim=True)
    #     return y
    
    def gumbel_softmax(self, logits, tau=1, hard=False, eps=1e-10, dim=-1):
        def _gen_gumbels():
            gumbels = -torch.empty_like(logits).exponential_().log()
            if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
                # to avoid zero in exp output
                gumbels = _gen_gumbels()
            return gumbels

        gumbels = _gen_gumbels()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret
    
    def forward(self, Q, K, V):
        '''
        Q: (bs, q_dim, 1)                                               #1,512,1
        K: (bs, n_select, dim), n_select could be num_obj or num_seg    #1,34,?
        V: (bs, n_select, n_frame_per_clip, obj_num, obj_dim)           #1,34,768
        '''
        # print(Q.shape, K.shape, V.shape)
        bs, n_select, _ = K.shape
        obj_num, obj_dim = V.shape[-2:]
        # from IPython.core.debugger import set_trace;
        # set_trace()
        v_shape = V.shape
        # V = V.view(bs, n_select, -1)

        # dbg.set_trace()

        Q = self.norm_Q(self.linear_Q(Q.squeeze(dim=-1)))  # [bs, dim, 1] -> [bs, dim]
        K = self.norm_K(self.linear_K(K.float()))  # [bs, numc, dim]
        V = V.float()

        logit_scale = 1
        x_logits = logit_scale * K @ Q.unsqueeze(dim=-1)
        x_logits = torch.softmax(x_logits.squeeze(dim=-1), dim=-1)

        selected_segs = []
        for _ in range(self.topk):
            # print(x_logits.shape)
            # selection_mask = self.sample_gumbel_softmax(x_logits, 1)
            # selection_mask = F.gumbel_softmax(x_logits, tau=1, dim=-1)
            selection_mask = self.gumbel_softmax(x_logits, tau=1, dim=-1)
            
            if torch.isnan(selection_mask).sum() or torch.isinf(selection_mask).sum():
                print(torch.isnan(selection_mask).sum(), torch.isinf(selection_mask).sum())
                # dbg.set_trace()
            selection_mask = selection_mask.unsqueeze(dim=1)
            if V.dim() == 3:
                selected_segs.append(
                    torch.matmul(selection_mask, V.view(bs, n_select, -1)))
            else:
                selected_segs.append(
                    torch.matmul(selection_mask, V.view(bs, n_select, -1)).view(bs, -1, obj_num, obj_dim))

        selected_segs = torch.cat(selected_segs, dim=1)  # [bs, topk * num_obj, CLIP_dim]

        return selected_segs
    
selector = Selector(topk=30, dim=768).cuda()
# dataset =============================================================================================================
class STAR(Dataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        self.args = args
        self.max_feats = args.max_feats
        self.features_dim = 768
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.split = split
        
        self.data = json.load(open(f'./data/star/STAR_{split}.json', 'r'))
        self.features = torch.load(f'./data/star/clipvitl140.pth')
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)'}
        self.qtype_mapping = {'Interaction': 1, 'Sequence': 2, 'Prediction': 3, 'Feasibility': 4}
        self.num_options = 4 # 4 kind of options
        
        self.clip_model, _ = clip_selector.load("ViT-L/14@336px")
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        self.qfeats = []
        if os.path.isfile(f'./data/star/qfeats_options_{split}.pth'):
            logging.info(f'Loading question features from "./data/star/qfeats_options_{split}.pth".')
            self.qfeats = torch.load(f'./data/star/qfeats_options_{split}.pth')
        else:
            logging.info(f"Encoding questions...")
            self.qfeats = []
            for i in range(len(self.data)):
                question_txt = self.data[i]['question']
                choices = ""
                
                
                for j in range(4): 
                    choices += (" " + self.data[i]['choices'][j]['choice'].lower())
                all_txt = question_txt + " The choices are:" + choices
                print(all_txt)

                question_clip = clip_selector.tokenize(all_txt)
                qfeat, _ = self.get_clip_txt_embedding(question_clip.cuda())
                self.qfeats.append(qfeat.cpu()[0])
            torch.save(self.qfeats, f'./data/star/qfeats_options_{split}.pth')
        
        logging.info(f"Num {split} data: {len(self.data)}")
    
    def _get_padding_id(self, text_id):
        padding_text_id = torch.zeros((len(text_id), self.max_seq_len), dtype=torch.int64) - 1
        for i, tid in enumerate(text_id):
            padding = self.max_seq_len - len(tid)
            if padding >= 0:
                padding_text_id[i, :len(tid)] = tid
            else:
                padding_text_id[i] = tid[:self.max_seq_len]
                logging.warning(f'max sequence length overflow: {len(tid)}')
        return padding_text_id
    
    def _get_text_token(self, text, answer):
        vqa_id, vqa_prefix_index, vqa_video_start = self.tokenizer.encode_vqa(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        vaq_id, vaq_prefix_index, vaq_video_start = self.tokenizer.encode_vaq(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        qav_id, qav_prefix_index                  = self.tokenizer.encode_qav(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        
        vqa_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in vqa_id]
        vaq_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in vaq_id]
        qav_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in qav_id]
        
        vqa_padding_text_id = self._get_padding_id(vqa_id)
        vaq_padding_text_id = self._get_padding_id(vaq_id)
        qav_padding_text_id = self._get_padding_id(qav_id)

        # label
        vqa_label = copy.deepcopy(vqa_padding_text_id)
        vqa_label[:, :vqa_prefix_index] = -1
        vqa_label_mask = vqa_label.ge(0)
        vqa_label[~vqa_label_mask] = 0
        vqa_label_mask = vqa_label_mask.float()
        
        vaq_label = copy.deepcopy(vaq_padding_text_id)
        vaq_label[:, :vaq_prefix_index] = -1
        vaq_label_mask = vaq_label.ge(0)
        vaq_label[~vaq_label_mask] = 0
        vaq_label_mask = vaq_label_mask.float()
        
        qav_label = torch.ones_like(qav_padding_text_id) * -1
                
        try:
            qav_label[:, qav_prefix_index:qav_prefix_index+self.max_feats] = torch.arange(self.max_feats)
        except:
            logging.error(f'qav_prefix_index: {qav_prefix_index}') # 99
            logging.error(f'qav_padding_text_id shape: {qav_padding_text_id.shape}') # torch.Size([1, 128])
            logging.error(f'qav_prefix_index + self.max_feats: {qav_prefix_index + self.max_feats}') # 129
            sys.exit()

        qav_label_mask = torch.zeros_like(qav_padding_text_id)
        qav_label_mask[:, qav_prefix_index] = 1
        qav_label_mask = qav_label_mask.float()
                
        # text mask
        vqa_text_mask = vqa_padding_text_id.ge(0)
        vqa_padding_text_id[~vqa_text_mask] = 0
        vaq_text_mask = vaq_padding_text_id.ge(0)
        vaq_padding_text_id[~vaq_text_mask] = 0
        qav_text_mask = qav_padding_text_id.ge(0)
        qav_padding_text_id[~qav_text_mask] = 0
        
        # video index
        vqa_video_index = torch.arange(vqa_prefix_index, vqa_prefix_index + self.max_feats)
        vaq_video_index = torch.arange(vaq_prefix_index, vaq_prefix_index + self.max_feats)
        qav_video_index = torch.arange(qav_prefix_index, qav_prefix_index + self.max_feats)
        
        
        text_id = {'vqa': vqa_padding_text_id, 'vaq': vaq_padding_text_id, 'qav': qav_padding_text_id}
        label = {'vqa': vqa_label, 'vaq': vaq_label, 'qav': qav_label}
        video_start = {'vqa': vqa_video_start, 'vaq': vaq_video_start, 'qav': qav_prefix_index}
        video_index = {'vqa': vqa_video_index, 'vaq': vaq_video_index, 'qav': qav_video_index}
        label_mask = {'vqa': vqa_label_mask, 'vaq': vaq_label_mask, 'qav': qav_label_mask}
        return text_id, label, video_start, video_index, label_mask

    def _get_text(self, idx):
        question = self.data[idx]["question"].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
            
        options = {x['choice_id']: x['choice'] for x in self.data[idx]['choices']}
        options = [options[i] for i in range(self.num_options)]
        
        if self.split == 'test':
            answer = -1
        else:
            answer = options.index(self.data[idx]['answer'])
        
        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        a_text = "Answer: The answer is "
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text, answer

    def _get_video(self, video_id, start, end):
        if video_id not in self.features:
            logging.warning(f'video_id: {video_id} not in features')
            video = torch.zeros(1, self.features_dim)
        else:
            video = self.features[video_id][start: end+1, :].float() # ts

        ## number of frames > 10, then sample to 10
        sample_range = 100
        
        if len(video) > sample_range:
            sampled = []
            for j in range(sample_range):
                sampled.append(video[(j * len(video)) // sample_range])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < sample_range:
            video_len = len(video) #FIXME
            # logging.warning(f'video_id: {video_id} video_len: {len(video)} < sample_range: {sample_range}')
            video = torch.cat([video, torch.full(((sample_range - len(video), self.features_dim)), 1e-15)], 0)
        else:
            video_len = self.max_feats

        return video, video_len
    
    def get_clip_txt_embedding(self, question):
        bsize = question.size(0)
        question_clip, word_clip = self.clip_model.encode_text(question.squeeze(dim=1))

        question_clip = question_clip / question_clip.norm(dim=-1, keepdim=True)   # [bsize, CLIP_dim]
        question_clip = question_clip.view(bsize, -1, 1).float()  # [bsize, 1, CLIP_dim]

        word_clip = word_clip / word_clip.norm(dim=-1, keepdim=True)   # [bsize, num_word, CLIP_dim]
        word_clip = word_clip.view(bsize, -1, 1).float()  # [bsize, num_word, CLIP_dim]
        return question_clip, word_clip

    def __getitem__(self, idx):
        vid = self.data[idx]['video_id']
        question_id = self.data[idx]['question_id']
        qtype = self.qtype_mapping[question_id.split('_')[0]]
        text, answer = self._get_text(idx)
        text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)
        
        sample_rate = 10
        
        start, end = round(self.data[idx]['start']*sample_rate), round(self.data[idx]['end']*sample_rate)
        video, video_len = self._get_video(f'{vid}', start, end)

        qfeat = self.qfeats[idx]

        return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                "video_index": video_index, "label_mask": label_mask, "qid": idx, "answer": answer,"qtype": qtype, "question_id": question_id, "qfeat": qfeat}

    def __len__(self):
        return len(self.data)
# load data -----------------------------------------------------------------------------------------------------------
def load_data(args, tokenizer, split='train'):
    args.num_options = 4 # STAR
    dataset = STAR(args=args, tokenizer=tokenizer, split=split)
    
    sampler = torch.utils.data.RandomSampler(dataset)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=batch_collate,
        pin_memory=args.pin_mem, 
        drop_last=False)

    return data_loader

def batch_collate(batch):
    bs = len(batch)
    vid = [batch[i]["vid"] for i in range(bs)]
    video = torch.stack([batch[i]["video"] for i in range(bs)])
    video_len = torch.tensor([batch[i]["video_len"] for i in range(bs)], dtype=torch.long)
    text = [batch[i]["text"] for i in range(bs)]
    qid = [batch[i]["qid"] for i in range(bs)]
    question_id = [batch[i]["question_id"] for i in range(bs)]
    qtype = torch.tensor([batch[i]['qtype'] for i in range(bs)])
    
    vqa_id = torch.stack([batch[i]['text_id']['vqa'] for i in range(bs)])
    vaq_id = torch.stack([batch[i]['text_id']['vaq'] for i in range(bs)])
    qav_id = torch.stack([batch[i]['text_id']['qav'] for i in range(bs)])
    text_id = {'vqa': vqa_id, 'vaq': vaq_id, 'qav': qav_id}
    
    vqa_label = torch.stack([batch[i]['label']['vqa'] for i in range(bs)])
    vaq_label = torch.stack([batch[i]['label']['vaq'] for i in range(bs)])
    qav_label = torch.stack([batch[i]['label']['qav'] for i in range(bs)])        
    label = {'vqa': vqa_label, 'vaq': vaq_label, 'qav': qav_label}
    
    vqa_video_start = [batch[i]["video_start"]['vqa'] for i in range(bs)]
    vaq_video_start = [batch[i]["video_start"]['vaq'] for i in range(bs)]
    qav_video_start = [batch[i]["video_start"]['qav'] for i in range(bs)]
    video_start = {'vqa': vqa_video_start, 'vaq': vaq_video_start, 'qav': qav_video_start}

    vqa_video_index = torch.stack([batch[i]["video_index"]['vqa'] for i in range(bs)])
    vaq_video_index = torch.stack([batch[i]["video_index"]['vaq'] for i in range(bs)])
    qav_video_index = torch.stack([batch[i]["video_index"]['qav'] for i in range(bs)])
    video_index = {'vqa': vqa_video_index, 'vaq': vaq_video_index, 'qav': qav_video_index}
    
    vqa_label_mask = torch.stack([batch[i]["label_mask"]['vqa'] for i in range(bs)])
    vaq_label_mask = torch.stack([batch[i]["label_mask"]['vaq'] for i in range(bs)])
    qav_label_mask = torch.stack([batch[i]["label_mask"]['qav'] for i in range(bs)])
    label_mask = {'vqa': vqa_label_mask, 'vaq': vaq_label_mask, 'qav': qav_label_mask}
    
    qfeat = [batch[i]["qfeat"] for i in range(bs)]
    qfeat = torch.stack(qfeat)

    answer = torch.tensor([batch[i]["answer"] for i in range(bs)])
    return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
            "video_index": video_index, "label_mask": label_mask, "qid": qid, "answer": answer, "qtype": qtype, "question_id": question_id, "qfeat": qfeat}
# model ===============================================================================================================
# Tokenizer ------------------------------------------------------------------------------------------------------------
class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        
        self.v_token_id = 15167
        self.q_token_id = 16492
        self.a_token_id = 22550
        self.nl_id = 13
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def encode_vqa(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the answer based on the video and question.\n"
        q_text = text['q_text']
        o_text = text['o_text']
        a_text = text['a_text']
     
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)

        s2 = q_text + o_text + a_text

        if split == 'train':
            s2 = s2 + answer_mapping[answer]
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2]
            prefix_index = t[0].index(self.a_token_id) + 5
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2)
            prefix_index = t[answer].index(self.a_token_id) + 5
        return t, prefix_index, video_start

    def encode_vaq(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the question based on the video and answer.\n"
        q_text = text['q_text'].strip()
        o_text = text['o_text']
        a_text = text['a_text']
        
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)
        
        s2 = o_text + a_text
        
        if split == 'train':
            s2 = s2 + answer_mapping[answer] + "\n" + q_text
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2]
            prefix_index = t[0].index(self.q_token_id) + 2
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v + "\n" + q_text) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2)
            prefix_index = t[answer].index(self.q_token_id) + 2
        return t, prefix_index, video_start
    
    def encode_qav(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the video based on the question and answer.\n"
        q_text = text['q_text']
        o_text = text['o_text']
        a_text = text['a_text']
        
        s1 = i_text + q_text + o_text + a_text
        
        if split == 'train':
            s1 = s1 + answer_mapping[answer] + "\n" + "Video:"
            t1 = [self.bos_id] + self.sp_model.encode(s1)
            t = [t1 + [-2 for _ in range(max_feats)] + [self.eos_id]]
            prefix_index = t[0].index(self.v_token_id) + 2
        else:
            t = []
            for k, v in answer_mapping.items():
                t1 = [self.bos_id] + self.sp_model.encode(s1 + v + "\n" + "Video:") + [-2 for _ in range(max_feats)] + [self.eos_id]
                t.append(t1)
            prefix_index = t[answer].index(self.v_token_id) + 2
        return t, prefix_index

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def encode_dvqa(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the answer based on the dialogue, video and question.\n"
        q_text = text['q_text']
        o_text = text['o_text']
        a_text = text['a_text']
        d_text = text['d_text']
     
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)
        
        prefix_i = video_start + max_feats + 1
        d1 = self.sp_model.encode(d_text)
        prefix_main = prefix_i + len(d1)

        s2 = q_text + o_text + a_text

        if split == 'train':
            s2 = s2 + answer_mapping[answer] 
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2]
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2)

        prefix_index = len(t[0]) - 4
        
        return t, prefix_index, video_start, prefix_i, prefix_main

    def encode_dvaq(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the question based on the dialogue, video and answer.\n"
        q_text = text['q_text'].strip()
        o_text = text['o_text']
        a_text = text['a_text']
        d_text = text['d_text']
        
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)
        
        prefix_i = video_start + max_feats + 1
        d1 = self.sp_model.encode(d_text)
        prefix_main = prefix_i + len(d1)

        s2 = o_text + a_text
        
        if split == 'train':
            s2 = s2 + answer_mapping[answer] + "\n" + q_text
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2]
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v + "\n" + q_text) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2)
        
        prefix_index = t[0].index(self.q_token_id) + 2
        
        return t, prefix_index, video_start, prefix_i, prefix_main
    
    def encode_dqav(self, text=None, max_feats=10, max_seq_len=128, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the video based on the dialogue, question and answer.\n"
        d_text = text['d_text']
        q_text = text['q_text']
        o_text = text['o_text']
        a_text = text['a_text']
        s1, s2, s3 = i_text, d_text, q_text + o_text + a_text

        t1 = [self.bos_id] + self.sp_model.encode(s1)
        t2 = self.sp_model.encode(s2)
        prefix_i, prefix_q = len(t1), len(t1) + len(t2)

        if split == 'train':
            t3 = self.sp_model.encode(s3 + answer_mapping[answer] + "\n" + "Video:")
            t = [t1 + t2 + t3 + [-2 for _ in range(max_feats)] + [self.eos_id]]
        else:
            t = []
            for k, v in answer_mapping.items():
                t3 = self.sp_model.encode(s3 + v + "\n" + "Video:") + [-2 for _ in range(max_feats)] + [self.eos_id]
                t.append(t1 + t2 + t3)
                
        prefix_index = len(t[0]) - max_feats - 1
        
        return t, prefix_index, prefix_i, prefix_q
# model ---------------------------------------------------------------------------------------------------------------
@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    adapter_len: int=10
    adapter_layer: int=30

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.max_feats = args.max_feats

        self.wq = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.gate1 = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))
        self.gate2 = torch.nn.Parameter(torch.ones(1, self.n_local_heads, 1, 1) * -args.bias)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, video_start=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_k = self.wk(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_v = self.wv(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            xk = torch.cat([adapter_k, xk], dim=1)
            xv = torch.cat([adapter_v, xv], dim=1)
            extra_mask = torch.zeros(1, 1, seqlen, adapter_len).to(mask)
            mask = torch.cat([extra_mask, mask], dim=-1)
        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        if adapter is not None:            
            adapter_scores = F.softmax(scores[..., :adapter_len].float(), dim=-1).type_as(xq) * self.gate1.tanh().half()
            if video_start is not None:
                vt_scores = scores[..., adapter_len:].clone()
                vt_scores[:, :, video_start + self.max_feats:, video_start:video_start + self.max_feats] = \
                    vt_scores[:, :, video_start + self.max_feats:, video_start:video_start + self.max_feats] + self.gate2.half()
                vt_scores = F.softmax(vt_scores.float(), dim=-1).type_as(xq)
            else:
                vt_scores = F.softmax(scores[..., adapter_len:], dim=-1)
            scores = torch.cat([adapter_scores, vt_scores], dim=-1)
        else:
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, video_start=None):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter, video_start)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, args):
        super().__init__()
        params.max_feats = args.max_feats
        params.bias = args.bias
        self.args = args
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.max_feats = args.max_feats


        self.tok_embeddings = Embedding(params.vocab_size, params.dim)

        self.adapter_query = Embedding(params.adapter_len * params.adapter_layer, params.dim)
        self.visual_proj = Linear(768, params.dim, bias=False)
        self.temporal_emb = Embedding(self.max_feats, params.dim)
        self.adapter_len = params.adapter_len
        self.adapter_layer = params.adapter_layer

        self.vqa_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.vaq_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.qav_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.inference_criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)

        self.video_label = torch.arange(1, self.max_feats)
        self.tau = args.tau

    def forward(self, data, inference=False):

        q_feat = data['qfeat'].cuda()
        video = data['video'].cuda()
        seg_feat = (video / video.norm(dim=-1, keepdim=True)).cuda()
        video = selector(q_feat, seg_feat, video)

        vqa_id, vaq_id, qav_id = data['text_id']['vqa'].cuda(), data['text_id']['vaq'].cuda(), data['text_id']['qav'].cuda()
        vqa_label, vaq_label, qav_label = data['label']['vqa'].cuda(), data['label']['vaq'].cuda(), data['label']['qav'].cuda()
        vqa_video_start, vaq_video_start, qav_video_index = data['video_start']['vqa'][0], data['video_start']['vaq'][0], data['video_index']['qav'].cuda()
        
        bsz, n_options, seqlen = vqa_id.shape
        vqa_id, vaq_id = vqa_id.reshape(-1, seqlen), vaq_id.reshape(-1, seqlen)
        vqa_label, vaq_label = vqa_label.reshape(-1, seqlen), vaq_label.reshape(-1, seqlen)
        vqa_label, vaq_label = vqa_label[:, 1:].flatten(), vaq_label[:, 1:].flatten()
        
        qav_id = qav_id.reshape(-1, seqlen)
        qav_label = qav_label.reshape(-1, seqlen)
        qav_video_mask = qav_label.ge(0)
        qav_label = qav_label[:, 1:].flatten()
        
        
        with torch.no_grad():
            vqa_h = self.tok_embeddings(vqa_id)
            
            if not inference:
                vaq_h = self.tok_embeddings(vaq_id)
            
            if not inference:
                qav_h = self.tok_embeddings(qav_id)
            
        freqs_cis = self.freqs_cis.to(vqa_h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=vqa_h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(vqa_h)
        start_pos = 0
        vaq_loss, qav_loss = torch.tensor([0]).cuda(), torch.tensor([0]).cuda()
        
        adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, self.params.dim).unsqueeze(1)
        _video_feature = self.visual_proj(video)
        if inference:
            _video_feature = _video_feature.unsqueeze(1).repeat(1, n_options, 1, 1).view(-1, _video_feature.shape[-2], _video_feature.shape[-1])
        video_feature = (_video_feature + self.temporal_emb.weight[None, :, :]).half()
        
        vqa_h = vqa_h.clone()
        vqa_h[:, vqa_video_start:vqa_video_start+self.max_feats] = video_feature

        
        if not inference:
            vaq_h = vaq_h.clone()
            vaq_h[:, vaq_video_start:vaq_video_start+self.max_feats] = video_feature
            
        if not inference:
            qav_h = qav_h * ~qav_video_mask[..., None]
            qav_h.scatter_add_(1, qav_video_index[..., None].repeat(1, 1, self.params.dim), video_feature)
        
        for i, layer in enumerate(self.layers[-1 * self.adapter_layer:]):
            vqa_h = layer(vqa_h, start_pos, freqs_cis, mask, adapter[i].half(), vqa_video_start)
            
            if not inference:
                vaq_h = layer(vaq_h, start_pos, freqs_cis, mask, adapter[i].half(), vaq_video_start)
            
            if not inference:
                qav_h = layer(qav_h, start_pos, freqs_cis, mask, adapter[i].half(), None)
        
        
        vqa_h = self.norm(vqa_h)
        vqa_output = self.output(vqa_h)
        vqa_output = vqa_output[:, :-1, :].reshape(-1, self.vocab_size)
        vqa_loss = self.vqa_criterion(vqa_output, vqa_label)
        
        if not inference:
            vaq_h = self.norm(vaq_h)
            vaq_output = self.output(vaq_h)
            vaq_output = vaq_output[:, :-1, :].reshape(-1, self.vocab_size)
            vaq_loss = self.vaq_criterion(vaq_output, vaq_label)
            
        if not inference:
            qav_h = self.norm(qav_h)
            qav_output = torch.bmm(qav_h[:, :-1].float(), _video_feature.transpose(1, 2).float()).reshape(-1, self.max_feats)
            qav_loss = self.qav_criterion(qav_output / self.tau, qav_label)
        
        if inference:
            logits = self.inference_criterion(vqa_output, vqa_label)
            logits = logits.reshape(bsz, n_options, -1)
            return logits
        else:
            return vqa_loss, vaq_loss, qav_loss
# generation ----------------------------------------------------------------------------------------------------------
class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompts: List[str], max_gen_len: int, temperature: float = 0.8, top_p: float = 0.95,) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.inference(None, tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
# lamma ---------------------------------------------------------------------------------------------------------------
def LLaMA_VQA(args, **kwargs):
    with open(f'{args.llama_model_path}{args.model}/params.json', "r") as f:
        params = json.loads(f.read())
    tokenizer = Tokenizer(model_path=f'{args.llama_model_path}/tokenizer.model')
    logging.info(f"Using model: {args.model}")
    
    checkpoints = (Path(args.llama_model_path) / args.model).glob("*.pth")
    checkpoints = sorted(checkpoints)
    
    loaded = []
    for x in checkpoints:
        logging.info(f"Loading checkpoint: {x}")
        loaded.append(torch.load(x, map_location="cpu"))
    
    if len(loaded) == 1:             # single model
        full_state_dict = loaded[0]
    else:                            # multi model
        full_state_dict = {}
        split_dims = {}
        
        def add_weight_with_split_dim(name, dim):
            if dim < 0:  # bcast without split
                full_state_dict[name] = loaded[0][name].clone()
            else:
                full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
            for x in loaded:
                del x[name]
            split_dims[name] = dim
        
        add_weight_with_split_dim("tok_embeddings.weight", 1)
        add_weight_with_split_dim("norm.weight", -1)
        add_weight_with_split_dim("output.weight", 0)
        for i in range(params["n_layers"]):
            logging.info(f"gathering layer {i} of {params['n_layers']}")
            layer_prefix = f"layers.{i}."
            bcast_names = ["attention_norm.weight", "ffn_norm.weight"]
            column_parallel_names = ["attention.wq.weight", "attention.wk.weight", "attention.wv.weight", "feed_forward.w1.weight", "feed_forward.w3.weight"]
            row_parallel_names = ["attention.wo.weight", "feed_forward.w2.weight"]
            for key in bcast_names:
                add_weight_with_split_dim(layer_prefix + key, -1)
            for key in column_parallel_names:
                add_weight_with_split_dim(layer_prefix + key, 0)
            for key in row_parallel_names:
                add_weight_with_split_dim(layer_prefix + key, 1)
    

    model_args: ModelArgs = ModelArgs(max_seq_len=args.max_seq_len, max_batch_size=32, adapter_len=args.adapter_len, adapter_layer=args.adapter_layer, **params)
    

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_llama_vqa = Transformer(model_args, args)
    torch.set_default_tensor_type(torch.FloatTensor)
    missing_keys, unexpected_keys = model_llama_vqa.load_state_dict(full_state_dict, strict=False)

    for name, param in model_llama_vqa.named_parameters():
        if ('gate' in name) or ('adapter' in name) or ('temporal_emb' in name) or ('visual_proj' in name):
            param.requires_grad = True
            param.data = param.data.float()
        else:
            param.requires_grad = False

    return model_llama_vqa
# learning rate scheduler =============================================================================================
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    elif args.constant_lr:
        lr = args.lr
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
# engine ==============================================================================================================
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n=n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, log_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}']
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            yield obj
            iter_time.update(time.time() - end)
            if i % log_freq == 0 or i == len(iterable) - 1:
                
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                logging.info(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self)))

            i += 1
            end = time.time()
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        logging.info('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / len(iterable)))
        
def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, loss_scaler, args=None, writer=None):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    log_freq = 100
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, log_freq, header)):

        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        vqa_loss, vaq_loss, qav_loss = model(data)

        loss = vqa_loss + vaq_loss + qav_loss
        loss_value = loss.item()
        vqa_loss_value = vqa_loss.item()
        vaq_loss_value = vaq_loss.item()
        qav_loss_value = qav_loss.item()

        if not math.isfinite(loss_value):
            logging.error("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(vqa_loss=vqa_loss_value)
        metric_logger.update(vaq_loss=vaq_loss_value)
        metric_logger.update(qav_loss=qav_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        writer.add_scalar('Training Loss', loss_value, global_step=epoch * len(data_loader) + data_iter_step)
        writer.add_scalar('VQA Loss', vqa_loss_value, global_step=epoch * len(data_loader) + data_iter_step)
        writer.add_scalar('VAQ Loss', vaq_loss_value, global_step=epoch * len(data_loader) + data_iter_step)
        writer.add_scalar('QAV Loss', qav_loss_value, global_step=epoch * len(data_loader) + data_iter_step)
        writer.add_scalar('Training Learning Rate', optimizer.param_groups[0]["lr"], global_step=epoch * len(data_loader) + data_iter_step)

    # gather the stats from all processes
    logging.info("Averaged stats: {}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def getCount(freq):
    count, total = freq[0], freq[1]
    return count / total if total != 0 else 0.0

def log_qtype(data, eval, metric_logger, args):
    ep = 1e-10
    

    qtype2id= {'In': 1, 'Seq': 2, 'Pre': 3, 'Feas': 4}


    q_freq = {i : [0., 0.] for i in qtype2id.values()}
    q_freq[0] = [0., 0.]
    for i, v in enumerate(eval):
        qt = data['qtype'][i].item()
        q_freq[qt][0] += v.item()
        q_freq[qt][1] += 1
        q_freq[0][0] += v.item()
        q_freq[0][1] += 1
    
    
    metric_logger.update(n=q_freq[1][1]+ep, In=getCount(q_freq[1]))
    metric_logger.update(n=q_freq[2][1]+ep, Seq=getCount(q_freq[2]))
    metric_logger.update(n=q_freq[3][1]+ep, Pre=getCount(q_freq[3]))
    metric_logger.update(n=q_freq[4][1]+ep, Feas=getCount(q_freq[4]))
    metric_logger.update(n=q_freq[0][1]+ep, Total=getCount(q_freq[0]))
         
def val_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, args=None, writer=None):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    log_freq = 100

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, log_freq, header)):
        answer = data['answer'].cuda()
        bsz = answer.shape[0]

        with torch.no_grad():
            logits = model(data, inference=True)
        
        count = (logits != 0).sum(-1)
        prediction = (logits.sum(-1) / count).argmin(-1)

        eval = (answer == prediction)
        acc = eval.sum().item() / bsz
        
        log_qtype(data, eval, metric_logger, args)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(n=bsz, acc=acc)
        
        writer.add_scalar('Validation Accuracy', acc, global_step=epoch * len(data_loader) + data_iter_step)
        writer.add_scalar('Validation Learning Rate', lr, global_step=epoch * len(data_loader) + data_iter_step)

    # gather the stats from all processes
    logging.info("Averaged stats: {}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# loss scaler =========================================================================================================
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
        
def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
# misc model ==========================================================================================================
def load_model(args, model, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        logging.info("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            logging.info("With optim & sched in scaler")
            
def save_model(args, epoch, model, optimizer, loss_scaler, name):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / (f'{name}.pth')]
        
        unfrozen_model = {}
        for n, p in model.named_parameters():
            if ('gate' in n) or ('adapter' in n) or ('temporal_emb' in n) or ('visual_proj' in n):
                unfrozen_model[n] = p

        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': unfrozen_model,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            torch.save(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)
# main ================================================================================================================
def main(args):
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, f'exp_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(args.output_dir)
    
    # save args to output dir
    with open(os.path.join(args.output_dir, "args.txt"), mode="w", encoding="utf-8") as f:
        f.write(json.dumps(args.__dict__, indent=4))

    logging.info('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    logging.info("args:\n{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    tokenizer = Tokenizer(model_path=f'{args.llama_model_path}./tokenizer.model')

    data_loader_train = load_data(args, tokenizer, split='train')
    data_loader_val = load_data(args, tokenizer, split='val')

    model = LLaMA_VQA(args).to(torch.device(args.device))
    
    logging.info("Model = %s" % str(model))
    logging.info("Model size = {:.3f} M".format(sum(p.numel() for p in model.parameters()) / 1024 / 1024))

    eff_batch_size = args.batch_size * args.accum_iter * 1
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    
    logging.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    logging.info("actual lr: %.2e" % args.lr)
    logging.info("accumulate grad iterations: %d" % args.accum_iter)
    logging.info("effective batch size: %d" % eff_batch_size)
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
    params = []
    for p in param_groups:
        params.append(p)
    params.append({'params':selector.parameters()})
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95))
    
    logging.info("Optimizer = %s" % str(optimizer))
    
    loss_scaler = NativeScalerWithGradNormCount()
    best_acc = 0.

    load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)

    logging.info("Start training for %d epochs" % args.epochs)
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(model, data_loader_train, optimizer, epoch, loss_scaler, args=args, writer=writer)
        val_stats = val_one_epoch(model, data_loader_val, optimizer, epoch, args=args, writer=writer)
        
        torch.save(selector.state_dict(), f'selector_epoch{epoch}.pt')

        if best_acc < val_stats['acc']:
            best_acc = val_stats['acc']
            
        model_name = f"checkpoint_epoch={epoch}_best_acc={best_acc}_acc={val_stats['acc']}"
        save_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name=model_name)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, **{f'val_{k}': v for k, v in val_stats.items()}}

        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    logging.info('Training time {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
    
    writer.close()


if __name__ == '__main__':
    
    args = get_args_parser()
    
    main(args)
