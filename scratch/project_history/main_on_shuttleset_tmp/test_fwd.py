"""Smoke-test script for BST_CG_AP forward pass. Path + n_class must track
whichever collated ablation is current; update both when a new ablation lands."""
import sys, os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

import torch, numpy as np
from model.bst import BST_CG_AP
ROOT = '/scratch/comp320a/ShuttleSet_data_une_merge_v1/dataset_npy_collated_between_2_hits_with_max_limits_seq_100_une_merge_v1_split_v2_dropunk/train'
m = BST_CG_AP(in_dim=72, n_class=29, seq_len=100).cuda()
hp = torch.from_numpy(np.load(f'{ROOT}/JnB_bone.npy')[:4]).float().cuda()
hp = hp.view(4, 100, 2, -1)
s = torch.from_numpy(np.load(f'{ROOT}/shuttle.npy')[:4]).float().cuda()
p = torch.from_numpy(np.load(f'{ROOT}/pos.npy')[:4]).float().cuda()
vl = torch.tensor([100,100,100,100]).cuda()
out = m(hp, s, p, vl)
print(f'output: nan={torch.isnan(out).any()}, shape={out.shape}, sample={out[0,:5]}')