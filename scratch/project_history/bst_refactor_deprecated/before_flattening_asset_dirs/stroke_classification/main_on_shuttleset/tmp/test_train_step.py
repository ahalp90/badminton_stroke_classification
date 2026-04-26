import sys, os

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

import torch, numpy as np
from model.bst import BST_CG_AP

root = '/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_collated_between_2_hits_with_max_limits_seq_100/train'
m = BST_CG_AP(in_dim=72, n_class=25, seq_len=100).cuda()
hp = torch.from_numpy(np.load(f'{root}/JnB_bone.npy')[:4]).float().cuda().view(4, 100, 2, -1)
s = torch.from_numpy(np.load(f'{root}/shuttle.npy')[:4]).float().cuda()
p = torch.from_numpy(np.load(f'{root}/pos.npy')[:4]).float().cuda()
vl = torch.tensor([100, 100, 100, 100]).cuda()
labels = torch.from_numpy(np.load(f'{root}/labels.npy')[:4]).long().cuda()

loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(m.parameters(), lr=5e-4)

out = m(hp, s, p, vl)
loss = loss_fn(out, labels)
print(f'loss: {loss.item()}')

optimizer.zero_grad()
loss.backward()
optimizer.step()

out2 = m(hp, s, p, vl)
loss2 = loss_fn(out2, labels)
print(f'loss after step: {loss2.item()}')