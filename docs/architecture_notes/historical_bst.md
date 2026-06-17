# Historical BST Reference

**Purpose.** This document preserves BST-origin and pre-Phase-2 content that has been excised from the active source tree. It is the canonical reference for the end-of-project report and diary, and for any reproduction or revert work.

Each section captures: where the content originally lived (file:line at the time of excision), why it was preserved through earlier phases, why it was removed, and the verbatim source as a fenced code block.

This file is **read-only history**. Do not edit excised content to reflect later changes; current state lives in the active source tree and in `bst_x_overview.md`.

---

## Status

Skeleton drafted 2026-04-26. Sections 1-6 captured in step 4 of the pre-phase-2 tidy (commit on branch `pre-phase-2-tidy`, parent SHA `342a573`). Source-code excision happens in step 5; this capture is byte-identical to the source as of `342a573`.

---

## 1. TemPose variant classes (deleted from `model/tempose.py` in step 5)

**Original location:** `src/bst_x/stroke_classification/model/tempose.py:156-667` (four classes).

**Why preserved through phase 0/1:** byte-identity reproduction with the upstream BST repo. TemPose is the BST paper's predecessor and its source was kept verbatim alongside `bst.py` so any backed-out comparison run could fall back to the original code.

**Why removed pre-phase-2:** none of the four standalone classes (`TemPose_V`, `TemPose_PF`, `TemPose_SF`, `TemPose_TF`) are imported anywhere outside `tempose.py`'s own `__main__` smoke check. `bst.py:28` only consumes the building-block utilities (`TCN`, `MLP`, `MLP_Head`, `FeedForward`, `TransformerEncoder`), which stay. TemPose is not a project baseline; the apples-to-apples preservation goal is served by the `BST_*` partials in `bst.py:433-437`.

### 1.1 `TemPose_V`

**Source:** `tempose.py:156-258` at SHA `342a573`.

```python
class TemPose_V(nn.Module):
    '''Similar to TemPose_TF in TemPose.'''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4
    ):
        super().__init__()

        self.project = nn.Linear(in_dim, d_model)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,  # JnB: (b, t, n, input_dim)
        video_len: Tensor  # video_len: (b)
    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)

        x = self.project(JnB)
        b, n, t, d = x.shape

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_tem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)

        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()

        x = self.mlp_head(x)
        return x
```

### 1.2 `TemPose_PF`

**Source:** `tempose.py:261-396` at SHA `342a573`.

```python
class TemPose_PF(nn.Module):
    '''For ablation studies.

    Equal to TemPose_TF without the shuttlecock trajectory
    or TemPose_V with the player positions.
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        '''`d_model` should be an even number.'''
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.project = nn.Linear(in_dim, d_model)

        # TCNs
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+2, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+2, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,  # JnB: (b, t, n, input_dim)
        pos: Tensor,  # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)

        x = self.project(JnB)
        b, n, t, d = x.shape

        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        # pos_top: (b, 2, t)
        # pos_bottom: (b, 2, t)

        # TCNs
        pos_top: Tensor = self.tcn_top(pos_top)
        pos_bottom: Tensor = self.tcn_bottom(pos_bottom)
        # pos_top: (b, d, t)
        # pos_bottom: (b, d, t)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        x_additional = torch.stack((pos_top, pos_bottom), dim=1)
        # x_additional: (b, 2, t, d)

        # Positions Fusion (PF)
        x = torch.cat((x, x_additional), dim=1)
        n += 2

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_tem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)

        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()

        x = self.mlp_head(x)
        return x
```

### 1.3 `TemPose_SF`

**Source:** `tempose.py:399-526` at SHA `342a573`.

```python
class TemPose_SF(nn.Module):
    '''For ablation studies.

    Equal to TemPose_TF without the player positions
    or TemPose_V with the shuttlecock trajectory.
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        '''`d_model` should be an even number.'''
        super().__init__()

        self.project = nn.Linear(in_dim, d_model)

        # TCNs
        tcn_channels = [d_model // 2, d_model]
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+1, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,  # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        video_len: Tensor  # video_len: (b)
    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)

        x = self.project(JnB)
        b, n, t, d = x.shape

        shuttle = shuttle.transpose(1, 2).contiguous()
        # shuttle: (b, 2, t)

        # TCN
        shuttle: Tensor = self.tcn_shuttle(shuttle)
        # shuttle: (b, d, t)

        shuttle = shuttle.transpose(1, 2).contiguous()
        x_additional = shuttle.unsqueeze(1)
        # x_additional: (b, 1, t, d)

        # Shuttlecock Fusion (SF)
        x = torch.cat((x, x_additional), dim=1)
        n += 1

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_tem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)

        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()

        x = self.mlp_head(x)
        return x
```

### 1.4 `TemPose_TF`

**Source:** `tempose.py:529-667` at SHA `342a573`.

```python
class TemPose_TF(nn.Module):
    '''Similar to TemPose_TF in TemPose.'''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        '''`d_model` should be an even number.'''
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.project = nn.Linear(in_dim, d_model)

        # TCNs
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+3, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,  # JnB: (b, t, n, input_dim)
        pos: Tensor,  # pos: (b, t, n, 2)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        video_len: Tensor  # video_len: (b)
    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)

        x = self.project(JnB)
        b, n, t, d = x.shape

        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()
        # pos_top: (b, 2, t)
        # pos_bottom: (b, 2, t)
        # shuttle: (b, 2, t)

        # TCNs
        pos_top: Tensor = self.tcn_top(pos_top)
        pos_bottom: Tensor = self.tcn_bottom(pos_bottom)
        shuttle: Tensor = self.tcn_shuttle(shuttle)
        # pos_top: (b, d, t)
        # pos_bottom: (b, d, t)
        # shuttle: (b, d, t)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)
        # x_additional: (b, 3, t, d)

        # Temporal Fusion (TF)
        x = torch.cat((x, x_additional), dim=1)
        n += 3

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_tem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)

        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()

        x = self.mlp_head(x)
        return x
```

---

## 2. Original BST `Hyp` namedtuple defaults

**Original location:** `src/bst_x/stroke_classification/main_on_shuttleset/bst_x_train.py:85-101` (commented-out block at SHA `342a573`).

**Why preserved:** the BST paper's published numbers are produced with these defaults; reproducing those numbers requires this exact configuration.

**Why removed:** by 2026-04-26 the active config has diverged enough that the commented-out block is misleading reference noise. The "no backwards-compat shims for unshipped code" principle applies; the values live here for the historical record.

```python
# hyp = Hyp(
#     n_epochs=1600,            # max epochs (will early-stop before this)
#     early_stop_n_epochs=300,  # stop if no F1 improvement for this many epochs
#     batch_size=128,
#     lr=5e-4,                  # initial learning rate (cosine-annealed during training)
#     warm_up_step=400,         # LR warmup steps before cosine decay begins
#     taxonomy='merged_25',      # key in TAXONOMIES: 'une_merge_v1', 'merged_25', 'raw_35', …
#     seq_len=100,              # frames per sample (must match data preprocessing)
#     pose_style='JnB_bone',   # 'J_only'=joints, 'JnB_bone'=joints+bones, 'Jn2B'=joints+2xbones
#     use_3d_pose=False,        # True for xyz keypoints, False for xy only
#     train_partial=1.0,        # fraction of training set to use (1.0 = all)
#     clips_csv=str(DEFAULT_CLIPS_CSV),
#     split_column='split_bst_baseline',
#     drop_unknown=False,
#     ablation_id=None,
# )
```

The original BST recipe also ran with `num_cycles=0.25` in the cosine scheduler call (`get_cosine_schedule_with_warmup`), so the LR barely decayed inside the useful training window. See section 3 for the retune rationale.

**Reproducing the BST paper's published numbers** today: branch off from a pre-tidy commit (most recent is `d4fd644`), use the `Hyp` defaults captured in this section, and run on the BST taxonomy (`merged_25`) with the BST split column (`split_bst_baseline`). Phase 1 demonstrated this reproduces the paper's headline numbers; the most recent phase-1 reproduction run is logged in the experiment manifest under `experiments/run_20260417_191851/`.

---

## 3. LR-schedule and aux-schedule retune rationale (excised from `bst_x_train.py`)

**Original location:** `src/bst_x/stroke_classification/main_on_shuttleset/bst_x_train.py:65-138` at SHA `342a573`.

**Why preserved through phase 1:** the dated rationale paragraphs ("LR-SCHEDULE RETUNE 2026-04-17", "AUX-SCHEDULE 2026-04-18") record decisions that materially affected the active config. Useful for ablation interpretation and for picking up the work after a long context gap.

**Why moved out:** the rationale belongs in a writeup-style document, not in the configuration block of the live training script. `bst_x_overview.md` carries the current-state distillation; the verbatim dated history lives here for the report.

### 3.1 LR-schedule retune (2026-04-17)

```text
# --------------------------------------------------------------------------
# LR-SCHEDULE RETUNE (2026-04-17)
# --------------------------------------------------------------------------
# The original BST recipe used n_epochs=1600, warm_up_step=400, patience=300,
# with a cosine scheduler at num_cycles=0.25 (see line below). In practice
# on ShuttleSet + merged_25, the TB log from run Apr17_13-04-35 showed:
#     - best F1_macro 0.8311 at epoch 41
#     - best F1_min   0.6250 at epoch 53
#     - best Val_loss 1.0032 at epoch 27
#     - training stopped at epoch 341 (patience fired)
# Convergence is well before epoch 60. At n_epochs=1600, num_cycles=0.25,
# the LR is still ~99.98% of peak at epoch 41, so cosine decay never
# actually bites during the useful window — the model just drifts under
# near-peak LR for hundreds of epochs and overfits.
# Retuned values (active block below) match the schedule to observed
# convergence: ~4-epoch warmup, short cosine that reaches 0 at the end,
# and patience tightened to 40 epochs to stay meaningful at the new
# length. n_epochs has since been shortened again (120 -> 80) to pair
# with the AUX-SCHEDULE block; see that block for rationale. The old
# values are preserved (commented) for easy revert.
```

### 3.2 Aux-schedule retune (2026-04-18)

```text
# --------------------------------------------------------------------------
# AUX-SCHEDULE (CG/AP warm-start-to-fade)
# --------------------------------------------------------------------------
# Hypothesis: Clean Gate and Aim Player are heuristics. They accelerate
# early convergence while the LR is hot, but they also constrain the
# representation the transformer backbone can learn. Annealing them out
# during training should let the backbone find a richer representation.
#
# Implementation: a scalar aux_factor in [0, 1] scales CG's dirt subtraction
# and blends AP's alpha multipliers toward pass-through (1.0). The schedule
# is cosine from epoch 1 (factor=1.0) to aux_fade_end_epoch (factor=0.0),
# then pinned at 0 for the rest of training (pure-backbone phase).
#
# Knobs (both below in the hyp block):
#   use_aux_schedule  -- False pins factor at 1.0 all run, reproducing the
#                        unscheduled BST_CG_AP baseline exactly.
#   aux_fade_end_epoch -- epoch at which the factor first reaches 0. Set
#                         well below n_epochs to guarantee a long pure-
#                         backbone tail. At 15 with n_epochs=80: warm-start
#                         is ~20% of training, then 65 epochs of pure-
#                         backbone at cooling LR — framed as warm-start
#                         then finetune. A prior 60/120 schedule put peaks
#                         inside the fade window (factor still 0.6-0.74 at
#                         pick time for 2 of 3 seeds), which diluted the
#                         signal into noise; the only seed that picked deep
#                         in the fade (factor ~0.1) showed the hoped-for
#                         min-F1 lift.
#
# CG and AP currently share one factor (set_schedule_factors passes the
# same value to both). If a later ablation wants independent fade timing,
# split into cg_fade_end_epoch and ap_fade_end_epoch and pass two values
# through aux_schedule_factor into set_schedule_factors.
#
# Test-time behaviour: the best checkpoint captures cg_factor and ap_factor
# as buffers in state_dict. task.test() runs forward with those restored
# values, so the final test metric reflects whichever factor was active at
# the best-F1 epoch. Nothing forces them to 0 or 1 at test time.
```

### 3.3 Cross-link

Current state: `docs/architecture_notes/bst_x_overview.md` (under "current LR + aux schedule").

---

## 4. Orphan dataset classes (deleted from `shuttleset_dataset.py` in step 5)

**Original location:** `src/bst_x/stroke_classification/preparing_data/shuttleset_dataset.py:142-633` at SHA `342a573`.

**Why preserved through phase 1:** these were experimental dataset variants for ablation studies that never made it onto the active path. Kept in case the ablation revived.

**Why removed:** no callers in active code; the variants assume `unknown_first=True` and break under `une_merge_v1_nosides` (the current default taxonomy).

### 4.1 `Dataset_npy` (lines 142-246)

**Purpose:** pre-flatten loader that read directly from the nested `{root}/{split}/{class}/` directory layout. Superseded by `Dataset_npy_collated`, which reads the master CSV's split and label columns.

```python
class Dataset_npy(Dataset):
    """Deprecated lazy per-clip Dataset.

    Expects the legacy nested ``{root_dir}/{set_name}/{class_folder}/`` layout
    (pre-Phase-2 pose writer output). Post-Phase-2 the per-clip npy files
    live flat under a single directory and split/label come from
    ``clips_master.csv`` at collation time, so this class's directory walk
    no longer matches what the writers produce. The only caller,
    ``Task.compare_pred_gt_on_specific_type`` in ``bst_x_train.py``, is a
    debug helper that is never invoked from the training or test paths.

    Use ``Dataset_npy_collated`` for BST training. If this helper ever
    needs to come back, rewrite it CSV-driven along the lines of
    ``collate_npy``.
    """
    def __init__(
        self,
        root_dir: Path,
        set_name: str,
        pose_style='J_only',
        seq_len=30,
        taxonomy: Taxonomy = TAXONOMY_RAW_35,
    ):
        import warnings
        warnings.warn(
            'Dataset_npy expects the legacy nested {split}/{class}/ layout '
            'and will not work against the post-Phase-2 flat per-clip dir. '
            'Use Dataset_npy_collated for training; rewrite CSV-driven if '
            'the compare_pred_gt_on_specific_type debug path ever returns.',
            DeprecationWarning, stacklevel=2,
        )
        super().__init__()
        assert set_name in ['train', 'val', 'test', 'test_specific'], 'Invalid set_name.'
        assert pose_style in ['J_only', 'JnB_interp', 'JnB_bone', 'Jn2B'], 'Invalid pose_style.'

        match set_name:
            case 'train':
                random_shift = RandomTranslation()
            case 'val' | 'test' | 'test_specific':
                random_shift = lambda x : x

        class_ls = taxonomy.class_list()

        # load .npy branch names
        data_branches = [str]
        labels = []

        if set_name != 'test_specific':
            target_dir = root_dir/set_name
            for typ in target_dir.iterdir():
                shots = sorted([str(s).replace('_pos.npy', '') for s in typ.glob('*_pos.npy')])
                data_branches += shots
                labels.append(np.full(len(shots), class_ls.index(typ.name), dtype=np.int64))
        else:
            data_branches = sorted([str(s).replace('_pos.npy', '') for s in root_dir.glob('*_pos.npy')])
            labels.append(np.full(len(data_branches), class_ls.index(root_dir.name), dtype=np.int64))

        self.data_branches = data_branches
        self.labels = np.concatenate(labels)

        self.pose_style = pose_style
        self.seq_len = seq_len
        self.random_shift = random_shift
        self.bone_pairs = get_bone_pairs(skeleton_format='coco')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        joints = np.load(self.data_branches[i]+'_joints.npy')
        # joints: (t, m, J, d)
        pos = np.load(self.data_branches[i]+'_pos.npy')
        # pos: (t, m, xy)
        shuttle = np.load(self.data_branches[i]+'_shuttle.npy')
        # shuttle: (t, xy)

        joints: np.ndarray = joints.astype(np.float32)
        pos: np.ndarray = pos.astype(np.float32)
        shuttle: np.ndarray = shuttle.astype(np.float32)

        joints, pos, shuttle, new_video_len = make_seq_len_same(self.seq_len, joints, pos, shuttle)

        self.random_shift(joints)

        match self.pose_style:
            case 'J_only':
                human_pose = joints
            case 'JnB_interp':
                human_pose = interpolate_joints(joints, self.bone_pairs)
            case 'JnB_bone':
                bones = create_bones(joints, self.bone_pairs)
                human_pose = np.concatenate((joints, bones), axis=-2)
            case 'Jn2B':
                joints = interpolate_joints(joints, self.bone_pairs)
                bones = create_bones(joints, self.bone_pairs)
                human_pose = np.concatenate((joints, bones), axis=-2)
            case _:
                NotImplementedError

        # human_pose: (t, m, pose, d)
        # pos: (t, m, xy)
        # shuttle: (t, xy)
        # new_video_len: int
        # label: int
        return (human_pose, pos, shuttle), new_video_len, self.labels[i]
```

### 4.2 `Dataset_npy_collated_one_side` (lines 351-420)

**Purpose:** ablation variant that exposed only one player's pose stream per sample. Used in the "single-side ablation" study during phase 0/1 exploration.

```python
class Dataset_npy_collated_one_side(Dataset):
    def __init__(
        self,
        root_dir: Path,
        set_name: str,
        pose_style='J_only',
        use_top_side=True,
        taxonomy: Taxonomy = TAXONOMY_RAW_35,
    ):
        '''Use Top / Bottom labels only. Thus, the length of the dataset becomes half.

        :param set_name: 'train', 'val', 'test'.
        :param pose_style: 'J_only', 'JnB_interp', 'JnB_bone', 'Jn2B'.

        Notice: There is no random translation here.

        .. warning::
            This class assumes the taxonomy has ``unknown_first=True`` so that
            'unknown' is at index 0, followed by all Top\\_ classes then all
            Bottom\\_ classes.  It uses ``unknown_i // 2`` to find the boundary
            between Top and Bottom label ranges.  Passing a taxonomy with
            ``unknown_first=False`` will produce incorrect label splits.
        '''
        super().__init__()

        assert set_name in ['train', 'val', 'test'], 'Invalid set_name.'
        assert pose_style in ['J_only', 'JnB_interp', 'JnB_bone', 'Jn2B'], 'Invalid pose_style.'
        assert taxonomy.unknown_first, (
            f'Dataset_npy_collated_one_side requires unknown_first=True, '
            f'but taxonomy {taxonomy.name!r} has unknown_first=False'
        )

        branch = root_dir/set_name

        self.human_pose = np.load(str(branch/f'{pose_style}.npy'))
        self.pos = np.load(str(branch/'pos.npy'))
        self.shuttle = np.load(str(branch/'shuttle.npy'))
        self.videos_len = np.load(str(branch/'videos_len.npy'))
        self.labels = np.load(str(branch/'labels.npy'))

        class_list = taxonomy.class_list()
        unknown_i = class_list.index('unknown')
        n_single = unknown_i // 2  # Top/Bottom boundary (assumes unknown is at index 0)
        if use_top_side:
            idx = (self.labels < n_single) | (self.labels == unknown_i)
            self.labels[self.labels == unknown_i] = n_single
        else:
            idx = self.labels >= n_single
            self.labels -= n_single

        self.human_pose = self.human_pose[idx]
        self.pos = self.pos[idx]
        self.shuttle = self.shuttle[idx]
        self.videos_len = self.videos_len[idx]
        self.labels = self.labels[idx]

        # J_only: (n, t, m, J, d)
        # JnB: (n, t, m, J+B, d)
        # Jn2B: (n, t, m, J+2B, d)
        # pos: (n, t, m, xy)
        # shuttle: (n, t, xy)
        # videos_len: (n)
        # labels: (n)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.human_pose[i], self.pos[i], self.shuttle[i]), \
                self.videos_len[i], self.labels[i]
```

### 4.3 `Dataset_npy_collated_single_pose` (lines 423-497)

**Purpose:** ablation variant that selected only the acting player's pose per sample (Top or Bottom according to label). Companion to `_one_side`.

```python
class Dataset_npy_collated_single_pose(Dataset):
    def __init__(
        self,
        root_dir: Path,
        set_name: str,
        pose_style='J_only',
        opposite_on_purpose=False,
        taxonomy: Taxonomy = TAXONOMY_RAW_35,
    ):
        '''Use Top / Bottom pose only. The length of the dataset is unchanged.

        :param set_name: 'train', 'val', 'test'.
        :param pose_style: 'J_only', 'JnB_interp', 'JnB_bone', 'Jn2B'.

        Notice: There is no random translation here.

        .. warning::
            This class assumes the taxonomy has ``unknown_first=True`` so that
            'unknown' is at index 0, followed by all Top\\_ classes then all
            Bottom\\_ classes.  It uses ``unknown_i // 2`` to find the boundary
            between Top and Bottom label ranges.  Passing a taxonomy with
            ``unknown_first=False`` will produce incorrect label splits.
        '''
        super().__init__()

        assert set_name in ['train', 'val', 'test'], 'Invalid set_name.'
        assert pose_style in ['J_only', 'JnB_interp', 'JnB_bone', 'Jn2B'], 'Invalid pose_style.'
        assert taxonomy.unknown_first, (
            f'Dataset_npy_collated_single_pose requires unknown_first=True, '
            f'but taxonomy {taxonomy.name!r} has unknown_first=False'
        )

        branch = root_dir/set_name

        self.human_pose = np.load(str(branch/f'{pose_style}.npy'))
        self.pos = np.load(str(branch/'pos.npy'))
        self.shuttle = np.load(str(branch/'shuttle.npy'))
        self.videos_len = np.load(str(branch/'videos_len.npy'))
        self.labels = np.load(str(branch/'labels.npy'))

        class_list = taxonomy.class_list()
        unknown_i = class_list.index('unknown')
        n_single = unknown_i // 2  # Top/Bottom boundary (assumes unknown is at index 0)

        top_i = (self.labels < n_single)
        bot_i = ~top_i & (self.labels != unknown_i)
        idx = top_i | bot_i

        if opposite_on_purpose:
            top_i, bot_i = bot_i, top_i

        human_pose = np.empty_like(self.human_pose[:, :, 0:1, :, :])
        human_pose[top_i] = self.human_pose[top_i, :, 0:1, :, :]
        human_pose[bot_i] = self.human_pose[bot_i, :, 1:2, :, :]
        self.human_pose = human_pose[idx]

        self.pos = self.pos[idx]
        self.shuttle = self.shuttle[idx]
        self.videos_len = self.videos_len[idx]
        self.labels = self.labels[idx]

        # J_only: (n, t, m, J, d)
        # JnB: (n, t, m, J+B, d)
        # Jn2B: (n, t, m, J+2B, d)
        # pos: (n, t, m, xy)
        # shuttle: (n, t, xy)
        # videos_len: (n)
        # labels: (n)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.human_pose[i], self.pos[i], self.shuttle[i]), \
                self.videos_len[i], self.labels[i]
```

### 4.4 Loader helpers

**`prepare_npy_loaders` (lines 500-531):**

```python
def prepare_npy_loaders(
    root_dir: Path,
    pose_style='Jn2B',
    seq_len=30,
    batch_size=128,
    use_cuda=True,
    num_workers=(0, 0, 0)
):
    train_set = Dataset_npy(root_dir, 'train', pose_style, seq_len)
    val_set = Dataset_npy(root_dir, 'val', pose_style, seq_len)
    test_set = Dataset_npy(root_dir, 'test', pose_style, seq_len)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=num_workers[0]
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        pin_memory=use_cuda,
        num_workers=num_workers[1]
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        pin_memory=use_cuda,
        num_workers=num_workers[2]
    )
    return train_loader, val_loader, test_loader
```

**`prepare_npy_collated_one_side_loaders` (lines 568-599):**

```python
def prepare_npy_collated_one_side_loaders(
    root_dir: Path,
    pose_style='Jn2B',
    use_top_side=True,
    batch_size=128,
    use_cuda=True,
    num_workers=(0, 0, 0)
):
    '''Notice that this one RandomTranslation is not used.'''
    train_set = Dataset_npy_collated_one_side(root_dir, 'train', pose_style, use_top_side)
    val_set = Dataset_npy_collated_one_side(root_dir, 'val', pose_style, use_top_side)
    test_set = Dataset_npy_collated_one_side(root_dir, 'test', pose_style, use_top_side)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=num_workers[0]
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        pin_memory=use_cuda,
        num_workers=num_workers[1]
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=num_workers[2]
    )
    return train_loader, val_loader, test_loader
```

**`prepare_npy_collated_single_pose_loaders` (lines 602-633):**

```python
def prepare_npy_collated_single_pose_loaders(
    root_dir: Path,
    pose_style='Jn2B',
    opposite_on_purpose=False,
    batch_size=128,
    use_cuda=True,
    num_workers=(0, 0, 0)
):
    '''Notice that this one RandomTranslation is not used.'''
    train_set = Dataset_npy_collated_single_pose(root_dir, 'train', pose_style, opposite_on_purpose)
    val_set = Dataset_npy_collated_single_pose(root_dir, 'val', pose_style, opposite_on_purpose)
    test_set = Dataset_npy_collated_single_pose(root_dir, 'test', pose_style, opposite_on_purpose)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=num_workers[0]
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        pin_memory=use_cuda,
        num_workers=num_workers[1]
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=num_workers[2]
    )
    return train_loader, val_loader, test_loader
```

---

## 5. `compare_pred_gt_on_specific_type` debug helper

**Original location:** `src/bst_x/stroke_classification/main_on_shuttleset/bst_x_train.py:706-733` at SHA `342a573`.

**Purpose:** Debug helper that loaded `Dataset_npy` and compared per-sample predictions against ground truth for a chosen stroke type. Output a Pandas DataFrame of mismatches with `Ball Round` / `Pred` / `GT` columns.

**Why preserved:** scratch debugging during phase 1 ablation passes.

**Why removed:** unreachable from the active run path; the only consumer of `Dataset_npy`. Removing it is what unblocks deleting `Dataset_npy` itself.

```python
def compare_pred_gt_on_specific_type(self, dir_path: Path):
    infer_ds = Dataset_npy(
        root_dir=dir_path,
        set_name='test_specific',
        pose_style=self.pose_style,
        seq_len=hyp.seq_len,
        taxonomy=self.taxonomy,
    )
    infer_loader = DataLoader(
        dataset=infer_ds,
        batch_size=hyp.batch_size,
    )

    pred, gt = test(self.net, infer_loader, self.device)
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()

    not_match = pred != gt
    class_ls = self.taxonomy.class_list()
    with pd.option_context('display.max_rows', None):
        df = pd.DataFrame(
            data={
                'Ball Round': [Path(e).stem for e in infer_ds.data_branches],
                'Pred': [class_ls[e] if b else '-' for e, b in zip(pred, not_match)],
                'GT': [class_ls[e] if b else '-' for e, b in zip(gt, not_match)]
            }
        )
        print(df)
```

---

## 6. `normalize_joints` upstream default (changed in `prepare_train_on_shuttleset.py` step 5)

**Original location:** `src/bst_x/stroke_classification/preparing_data/prepare_train_on_shuttleset.py:160-175` at SHA `342a573`.

**Original signature:** `normalize_joints(..., center_align=False)` (matching upstream BST verbatim).

**Production behaviour:** every active caller passes `joints_center_align=True`. The signature default has been misleading since the refactor.

**Change in step 5:** the signature default flips to `True`. The apologia paragraph below was the in-source explanation of the mismatch.

```python
"""
- `arr`: (m, J, 2), m=2.
- `bbox`: (m, 4), m=2.

Output: (m, J, 2), m=2.

Signature defaults preserved verbatim from BST upstream for canonical
accuracy. The CLI invocation in ``main()`` below overrides
``center_align`` to True (matches BST upstream's own CLI default;
committed ShuttleSet extracts were produced with this override).
``v_height=None`` is canonical at both layers: the signature default
and the CLI call agree, so no flip happens there.
"""
```

For reproducing the BST paper's published numbers, use `center_align=False` and the CLI default of `True` (i.e. revert the step-5 default flip and pass `center_align=False` to any direct caller).

---

## 7. Migration anchors and task-anchored comments removed from `bst_x_train.py` (step 10)

**Original locations at SHA `342a573`:**

- `bst_x_train.py:1-2`:
  ```python
  # Consolidated BST training script for ShuttleSet
  # Replaces: bst_main.py, bst_main_summary_writer.py, bst_backbone_main.py
  ```
- `bst_x_train.py:53-57`: refactor cross-ref to `docs/architecture_notes/completed_general_refactors/dir_flatten_refactor.md` (block comment preserved as-is below).
- `bst_x_train.py:151`:
  ```python
  use_aux_schedule=True,    # Aggressive CG/AP annealing — matches preferred config from run_20260418_151139.
  ```

**Why preserved:** lineage record during the multi-step phase-1 consolidation.

**Why removed:** the migrations are done; the run id reference rots as new ablation runs supersede it.

The `completed_general_refactors/` directory at `docs/architecture_notes/completed_general_refactors/` still holds the long-form refactor notes; this file just records that the in-source pointers to it have been removed in step 10.

---

## 8. Project-history relocations (step 3, no source-code change)

**2026-04-26:** moved out of `src/` into `scratch/project_history/`:

- `src/bst_x/deprecated/` → `scratch/project_history/bst_refactor_deprecated/`
- `src/bst_x/ShuttleSet/deprecated/` → `scratch/project_history/shuttleset_deprecated/`
- `src/bst_x/stroke_classification/main_on_shuttleset/tmp/` → `scratch/project_history/main_on_shuttleset_tmp/`

The relocated trees include:
- The original BST author's pre-phase-1 scripts (`gen_my_dataset.py`, `get_each_class_total.py`, etc.).
- The pre-flatten snapshot of `pipeline/` and `stroke_classification/` (`before_flattening_asset_dirs/`).
- Phase-0 historical documentation (`outdated_bst_repo_reusability_assessment.md`, `outdated_bst_models_refactor.md`, `outdated_pipeline_build.md`, `historical_README_bst_original.md`, `historical_predecessor_analysis_summary.md`).
- The `tmp/` smoke tests (`test_dataloader.py`, `test_fwd.py`, `test_train_step.py`).

Original locations are recorded in `scratch/project_history/README.md`.

---

## Cross-references

- `docs/architecture_notes/bst_x_overview.md` — current Architecture 1 state and recent decision history.
- `docs/architecture_notes/pipeline_context_notes.md` — pipeline-area excisions (separate from BST-core).
- `docs/architecture_notes/pre_phase_2_review_2026-04-26.md` — review that drove the tidy pass.
- `docs/architecture_notes/pre_phase_2_tidy_plan.md` — execution plan for the tidy pass.
- `.claude/project_overview.md` — project handover document.

---

## Maintenance

Append new sections at the end as further excisions happen. Do not edit existing sections to reflect later changes; spawn new sections cross-linked to the old ones if behaviour is later restored or revised.
