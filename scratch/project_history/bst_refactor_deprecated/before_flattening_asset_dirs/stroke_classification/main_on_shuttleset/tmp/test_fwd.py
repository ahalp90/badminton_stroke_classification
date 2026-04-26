import sys, os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

import torch, numpy as np
from model.bst import BST_CG_AP
m = BST_CG_AP(in_dim=72, n_class=25, seq_len=100).cuda()
hp = torch.from_numpy(np.load('/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_collated_between_2_hits_with_max_limits_seq_100/train/JnB_bone.npy')[:4]).float().cuda()
hp = hp.view(4, 100, 2, -1)
s = torch.from_numpy(np.load('/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_collated_between_2_hits_with_max_limits_seq_100/train/shuttle.npy')[:4]).float().cuda()
p = torch.from_numpy(np.load('/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_collated_between_2_hits_with_max_limits_seq_100/train/pos.npy')[:4]).float().cuda()
vl = torch.tensor([100,100,100,100]).cuda()
out = m(hp, s, p, vl)
print(f'output: nan={torch.isnan(out).any()}, shape={out.shape}, sample={out[0,:5]}')