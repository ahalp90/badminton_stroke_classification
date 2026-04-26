import sys, os

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

import torch
from pathlib import Path
from preparing_data.shuttleset_dataset import prepare_npy_collated_loaders, RandomTranslation_batch, get_bone_pairs, \
    POSE_BONE_MULTIPLIER
from model.bst import BST_CG_AP

root = Path('../preparing_data/ShuttleSet_data_merged_25/dataset_npy_collated_between_2_hits_with_max_limits_seq_100')
train_loader, _, _ = prepare_npy_collated_loaders(root, pose_style='JnB_bone', batch_size=128, use_cuda=True,
                                                  num_workers=(0, 0, 0))

m = BST_CG_AP(in_dim=72, n_class=25, seq_len=100).cuda()
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(m.parameters(), lr=5e-4)
shift_fn = RandomTranslation_batch()
n_bones = len(get_bone_pairs())

for i, ((hp, pos, shuttle), vl, labels) in enumerate(train_loader):
    hp = hp.cuda()
    pos = pos.cuda()
    shuttle = shuttle.cuda()
    vl = vl.cuda()
    labels = labels.cuda()

    joints = hp[:, :, :, :-n_bones, :].contiguous()
    bones = hp[:, :, :, -n_bones:, :]
    joints = shift_fn(joints)
    hp = torch.cat([joints, bones], dim=-2)
    hp = hp.view(*hp.shape[:-2], -1)

    out = m(hp, shuttle, pos, vl)
    loss = loss_fn(out, labels)

    if torch.isnan(loss):
        print(f'NaN at batch {i}!')
        print(f'  hp: nan={torch.isnan(hp).any()}, inf={torch.isinf(hp).any()}')
        print(f'  shuttle: nan={torch.isnan(shuttle).any()}')
        print(f'  pos: nan={torch.isnan(pos).any()}')
        print(f'  out: nan={torch.isnan(out).any()}, max={out.max()}, min={out.min()}')
        print(f'  labels: min={labels.min()}, max={labels.max()}')
        print(f'  vl: min={vl.min()}, max={vl.max()}, dtype={vl.dtype}')
        break

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'batch {i}: loss={loss.item():.4f}')
    if i >= 5:
        print('First 6 batches OK')
        break