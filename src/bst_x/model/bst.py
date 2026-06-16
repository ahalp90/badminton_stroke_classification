# Original BST by Jing-Yuan Chang
# Refactored: consolidated 4 variant classes into 1 configurable class
#
# PyTorch notes for TensorFlow users:
#   nn.Module    = tf.keras.Model (base class for all models/layers)
#   nn.Parameter = a learnable tensor, like tf's self.add_weight() but standalone
#   forward()    = the __call__/call() method — PyTorch calls it when you do model(x)
#   .to(device)  = moves tensor to GPU/CPU — TF handles this automatically
#   .contiguous()= ensures tensor is stored in contiguous memory after transpose/permute
#                  (needed before .view(); harmless no-op if already contiguous)

import torch
from torch import nn, Tensor
from positional_encodings.torch_encodings import PositionalEncoding1D
from functools import partial  # partial(fn, arg=val) creates a new fn with some args pre-filled

# Building blocks defined in tempose.py:
#   TCN                = temporal convolution network (dilated 1D convs for sequence features)
#   MLP                = Linear -> GELU -> Dropout -> Linear
#   MLP_Head           = LayerNorm -> MLP (final classification layer)
#   FeedForward        = MLP + Dropout (used inside transformer layers)
#   TransformerEncoder = stack of self-attention TransformerLayers
from model.tempose import TCN, FeedForward, MLP, MLP_Head, TransformerEncoder


class MultiHeadCrossAttention(nn.Module):
    """Cross-attention: x1 asks questions (queries), x2 provides answers (keys+values).
    Unlike self-attention where one input attends to itself, cross-attention lets one
    sequence attend to a different sequence — here, a player attending to the shuttle.

    Key dimensions:
        d_model = feature size of input/output (e.g. 100)
        d_head  = feature size per attention head (e.g. 128)
        n_head  = number of parallel attention heads (e.g. 6)
        d_cat   = d_head * n_head = total projection size across all heads
    """
    def __init__(self, d_model, d_head, n_head, drop_p) -> None:
        super().__init__()  # PyTorch equivalent of super().__init__() in tf.keras.Model
        d_cat = d_head * n_head

        self.h = n_head
        # Queries come from x1, keys+values come from x2 (this is what makes it "cross")
        self.to_q = nn.Linear(d_model, d_cat, bias=False)
        self.to_kv = nn.Linear(d_model, d_cat * 2, bias=False)  # *2 because K and V are packed together
        self.scale = d_head**-0.5  # 1/sqrt(d_head) — standard attention scaling factor

        self.attend = nn.Sequential(  # nn.Sequential = tf.keras.Sequential
            nn.Softmax(dim=-1),
            nn.Dropout(drop_p)  # This shouldn't be inplace.
        )

        # Project multi-head output back to d_model, or skip if dimensions already match
        # nn.Identity() = a no-op layer that passes input through unchanged
        self.tail = nn.Sequential(
            nn.Linear(d_cat, d_model),
            nn.Dropout(drop_p, inplace=True)
        ) if n_head != 1 or d_cat != d_model else nn.Identity()

    def forward(self, x1: Tensor, x2: Tensor, mask: Tensor = None):
        # x1, x2: (b, t, d_model)
        q: Tensor = self.to_q(x1)   # queries from x1
        kv: Tensor = self.to_kv(x2) # keys+values from x2
        b, t, _ = q.shape

        # Reshape to separate attention heads: (b, t, d_cat) -> (b, h, t, d_head)
        # .view() = .reshape() but requires contiguous memory (faster, no copy)
        # .transpose(1,2) = swap dims 1 and 2, like tf.transpose(perm=[0,2,1,3])
        q = q.view(b, t, self.h, -1).transpose(1, 2)
        # .chunk(2, dim=-1) = split last dim in half -> (K, V) tuple
        kv = kv.view(b, t, self.h, -1).chunk(2, dim=-1)
        k, v = map(lambda ts: ts.transpose(1, 2), kv)
        # q, k, v: (b, h, t, d_head)

        # Attention scores: Q @ K^T / sqrt(d_head)
        # @ = matrix multiply (same as tf.matmul or np.matmul)
        dots: Tensor = (q.contiguous() @ k.transpose(-1, -2).contiguous()) * self.scale
        # dots: (b, h, t, t) — attention score for every (query_pos, key_pos) pair
        if mask is not None:
            # mask: (b, t) — True for real frames, False for padding
            mask = mask.view(b, 1, 1, t)
            # Set padded positions to -inf so softmax gives them zero weight
            dots = dots.masked_fill(~mask, -torch.inf)

        coef = self.attend(dots)  # softmax -> dropout
        # Weighted sum of values by attention coefficients
        attention: Tensor = coef @ v.contiguous()
        # attention: (b, h, t, d_head)

        # Merge heads back: (b, h, t, d_head) -> (b, t, h*d_head)
        out = attention.transpose(1, 2).reshape(b, t, -1)
        # out: (b, t, h*d_head)
        out = self.tail(out)  # project back to d_model
        return out  # (b, t, d_model)


class CrossTransformerLayer(nn.Module):
    """One transformer layer using cross-attention (x1 attends to x2) + feed-forward.
    Used here so each player's representation can attend to the shuttle trajectory.

    NOTE: The residual connection is only around the FFN, NOT around the cross-attention.
    A standard transformer would do: x = cross_attn(x1, x2) + x1 (residual back to query).
    Here the cross-attention output *replaces* x1 entirely. This likely anchors the output
    in shuttle-space rather than mixing back in raw player pose features, consistent with
    BST's thesis that shuttle trajectory is a better stroke descriptor than player pose.
    Under investigation with the original author — could also be a bug."""
    def __init__(self, d_model, d_head, n_head, hd_mlp, drop_p) -> None:
        super().__init__()
        self.layer_norm1_x1 = nn.LayerNorm(d_model)  # nn.LayerNorm = tf.keras.layers.LayerNormalization
        self.layer_norm1_x2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, d_head, n_head, drop_p)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_model, hd_mlp, drop_p)

    def forward(self, x1: Tensor, x2: Tensor, mask=None):
        x1 = self.layer_norm1_x1(x1)
        x2 = self.layer_norm1_x2(x2)
        x = self.cross_attn(x1, x2, mask)  # x1 queries, x2 provides context
        z = self.layer_norm2(x)
        x = self.ff(z) + x  # residual connection (output = transform(x) + x)
        return x


class BST(nn.Module):
    '''Unified BST (Badminton Stroke-type Transformer) with optional modules.

    Three boolean flags control which optional modules are active:
      use_ppf : Pose Position Fusion — fuses court xy into skeleton features before TCN
      use_cg  : Clean Gate — subtracts shared player noise from shuttle representation
      use_ap  : Aim Player — weights player contributions by cosine similarity to shuttle

    Original variant mapping:
      BST_0     = BST(use_ppf=False, use_cg=False, use_ap=False)
      BST (PPF) = BST(use_ppf=True,  use_cg=False, use_ap=False)
      BST_CG    = BST(use_ppf=True,  use_cg=True,  use_ap=False)
      BST_AP    = BST(use_ppf=True,  use_cg=False, use_ap=True)
      BST_CG_AP = BST(use_ppf=True,  use_cg=True,  use_ap=True)
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        use_ppf=True, use_cg=False, use_ap=False
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        # Store flags for use in forward()
        self.use_ppf = use_ppf
        self.use_cg = use_cg
        self.use_ap = use_ap

        # --- Optional: Pose Position Fusion (PPF) ---
        # Projects 2D court positions to in_dim and fuses with skeleton via multiplication
        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p) if use_ppf else None

        # --- Always created: TCN feature extractors ---
        # TCN = temporal convolution network with dilated 1D convolutions.
        # Like a 1D CNN that operates along the time axis with increasing receptive field.
        # in_dim -> [d_model, d_model] means two conv layers, both outputting d_model channels.
        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # --- Always created: Temporal Transformer (processes each stream independently) ---
        # "Class token" (CLS): a learnable vector prepended to the sequence that the
        # transformer uses as a summary. After attention, position 0 (the CLS token)
        # contains a learned summary of the entire sequence. This is the standard
        # ViT/BERT trick — instead of pooling over all positions, you read one token.
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        # Positional embeddings: added to input so the transformer knows frame order
        # (transformers have no built-in notion of sequence position unlike LSTMs)
        # Intentionally learnable (nn.Parameter): initialised with sinusoidal values in
        # init_weights(), but gradients are enabled so the model can fine-tune positions
        # during training. This gives a sensible starting point without constraining it.
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))  # 1+ for CLS
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # --- Always created: Cross Transformer (player attends to shuttle) ---
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # --- Always created: Interactional Transformer (models cross-player dynamics) ---
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # --- Optional: Aim Player (AP) ---
        # Cosine similarity between player and shuttle representations determines alpha weighting
        self.cos_sim = nn.CosineSimilarity() if use_ap else None

        # --- Optional: Clean Gate (CG) ---
        # MLP learns what shared player noise to subtract from shuttle representation
        self.mlp_clean = MLP(d_model, d_model, d_model, drop_p) if use_cg else None

        # --- MLP Head ---
        # AP without CG drops shuttle from head input (2*d_model vs 3*d_model)
        head_dim = d_model * 2 if (use_ap and not use_cg) else d_model * 3
        self.mlp_head = MLP_Head(head_dim, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        # Scheduling factors for CG/AP warm-start, scalar in [0, 1].
        # Buffers (not Parameters): not learned, travel with .to(device), saved in state_dict.
        # Overwritten per epoch by the trainer via set_schedule_factors().
        self.register_buffer('cg_factor', torch.tensor(1.0))
        self.register_buffer('ap_factor', torch.tensor(1.0))

        self.init_weights()

    def set_schedule_factors(self, cg_factor: float, ap_factor: float):
        """Overwrite scheduling factors for Clean Gate and Aim Player.

        Called by the training loop once per epoch to implement a warm-start
        schedule: factors start at 1.0 (full CG/AP), decay toward 0.0 so the
        transformer backbone increasingly stands on its own.

        :param cg_factor: scalar in [0, 1]; scales dirt subtraction in CG.
        :param ap_factor: scalar in [0, 1]; blends AP alpha toward pass-through.
        :return: None. Mutates buffers in place.
        """
        # .fill_() overwrites in place and preserves device/dtype after .to('cuda').
        self.cg_factor.fill_(cg_factor)
        self.ap_factor.fill_(ap_factor)

    @torch.no_grad()  # disable gradient tracking — this is init, not training
    def init_weights(self):
        """Initialize positional encodings and learnable tokens.
        In TF, kernel_initializer handles this; PyTorch requires explicit init."""
        # Sinusoidal positional encodings: give the transformer a sense of frame order
        p_enc_1d_model = PositionalEncoding1D(self.d_model)

        # .copy_() = in-place overwrite of the parameter's values
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Small random init for class tokens
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        # .apply() walks every sub-module and calls the given function on each
        # (like tf.keras.Model recursively visiting all layers)
        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        """Called on every sub-module by .apply() above. Sets initial weight values.
        Xavier init keeps signal variance stable through deep networks."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,       # (b, t, n, input_dim) — skeleton joint/bone features per player
        shuttle: Tensor,    # (b, t, 2) — shuttle xy coordinates per frame
        pos: Tensor = None, # (b, t, n, 2) — player court xy positions (required if use_ppf)
        video_len: Tensor = None  # (b,) — real frame count per sample (rest is zero-padding)
    ):
        """Forward pass. Shape key: b=batch, t=timesteps, n=players(2), d=d_model(100).
        Pipeline: TCN -> Temporal Transformer -> Cross Transformer -> Interactional Transformer -> Head
        """
        b, t, n, in_dim = JnB.shape
        # Rearrange for 1D conv: PyTorch Conv1d expects (batch, channels, length)
        # .permute() = reorder dimensions (like tf.transpose with perm=)
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        # JnB: (b*n, in_dim, t) — both players stacked in batch dim for parallel TCN

        # ====================================================================
        # [PPF] Pose Position Fusion: modulate skeleton features by court position
        # ====================================================================
        if self.use_ppf:
            pos = self.mlp_positions(pos)
            # pos: (b, t, n, in_dim)
            pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
            # pos_impact: (b*n, in_dim, t)
            JnB = JnB * pos_impact + JnB
            # Multiplicative fusion with residual: JnB * (1 + pos_impact)

        # ====================================================================
        # TCN: extract temporal features from pose and shuttle
        # ====================================================================
        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)
        # JnB: (b, n, t, d_model)

        shuttle = shuttle.transpose(1, 2).contiguous()
        # shuttle: (b, 2, t)
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        # shuttle: (b, 1, t, d_model)

        x = torch.cat((JnB, shuttle), dim=1)
        # x: (b, n+1, t, d_model) where n+1 = 3 (p1, p2, shuttle)
        _, n, _, d = x.shape

        # ====================================================================
        # Temporal Transformer: each stream (p1, p2, shuttle) processed independently
        # ====================================================================
        # Prepend a learnable CLS token to each stream's sequence.
        # .expand() = broadcast to larger size without copying memory (like tf.broadcast_to)
        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        # Concatenate CLS token at position 0, then add positional embeddings
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem
        # x: (b*n, 1+t, d_model) — "1+" because CLS token is prepended

        # Build padding mask: True for real frames + CLS, False for zero-padded frames.
        # This prevents the transformer from attending to meaningless padding positions.
        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        # repeat_interleave: duplicate each mask row n times (one per stream: p1, p2, shuttle)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)  # self-attention across time within each stream
        x = x.view(b, n, 1+t, d)
        # x: (b, 3, 1+t, d_model) — 3 streams: [player1, player2, shuttle]

        # ====================================================================
        # Split the 3 streams back apart and extract their CLS tokens
        # ====================================================================
        # .chunk(3, dim=1) = split dim 1 into 3 equal parts (like tf.split)
        # .squeeze(1) = remove the now-singleton dim 1
        p1, p2, shuttle = map(lambda ts: ts.squeeze(1), x.chunk(3, dim=1))
        # p1, p2, shuttle: each (b, 1+t, d_model)

        # Extract CLS tokens (position 0) — these are learned summaries of each stream
        p1_cls, p2_cls, shuttle_cls = \
            p1[:, 0].contiguous(), p2[:, 0].contiguous(), shuttle[:, 0].contiguous()
        # *_cls: (b, d_model) — one summary vector per stream per batch item

        # Remaining sequence positions (frames 1..t), with fresh positional embeddings
        p1 = p1[:, 1:].contiguous() + self.embedding_cross
        p2 = p2[:, 1:].contiguous() + self.embedding_cross
        shuttle = shuttle[:, 1:].contiguous() + self.embedding_cross
        # p1, p2, shuttle: (b, t, d_model)

        # ====================================================================
        # Cross Transformer: player-shuttle interaction
        # ====================================================================
        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)
        # p1_shuttle, p2_shuttle: (b, t, d_model)

        # ====================================================================
        # Interactional Transformer: cross-player modelling
        # ====================================================================
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter
        # p1_shuttle, p2_shuttle: (b, 1+t, d_model)

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()
        # p1_shuttle_cls, p2_shuttle_cls: (b, d_model)

        # ====================================================================
        # Combine temporal and interactional class tokens per player
        # ====================================================================
        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls
        # p1_conclusion, p2_conclusion: (b, d_model)

        # ====================================================================
        # [AP] Aim Player: weight player contributions by shuttle similarity
        # ====================================================================
        if self.use_ap:
            p1_shuttle_sim = self.cos_sim(p1_shuttle_cls, shuttle_cls)
            p2_shuttle_sim = self.cos_sim(p2_shuttle_cls, shuttle_cls)
            alpha: Tensor = (p1_shuttle_sim - p2_shuttle_sim + 2) / 4
            # alpha: (b,) in [0, 1] — higher means p1 is more relevant
            alpha = alpha.unsqueeze(1)
            # alpha: (b, 1)
            # Warm-start schedule: blend each multiplier toward 1.0 as ap_factor -> 0.
            # ap_factor=1 recovers original AP; ap_factor=0 makes both multipliers exactly 1
            # (pass-through — p1_conclusion and p2_conclusion are unchanged).
            eff_alpha_p1 = self.ap_factor * alpha + (1 - self.ap_factor)
            eff_alpha_p2 = self.ap_factor * (1 - alpha) + (1 - self.ap_factor)
            p1_conclusion = eff_alpha_p1 * p1_conclusion
            p2_conclusion = eff_alpha_p2 * p2_conclusion

        # ====================================================================
        # [CG] Clean Gate: remove shared player noise from shuttle
        # ====================================================================
        if self.use_cg:
            info_need_clean = torch.minimum(p1_shuttle_cls, p2_shuttle_cls)
            dirt = self.mlp_clean(info_need_clean)
            # Warm-start schedule: cg_factor=1 applies full CG; cg_factor=0 bypasses the subtraction.
            shuttle_cls = shuttle_cls - self.cg_factor * dirt

        # ====================================================================
        # MLP Head: final classification
        # ====================================================================
        # AP without CG drops shuttle (shuttle info is encoded in alpha weighting)
        if self.use_ap and not self.use_cg:
            x = torch.cat((p1_conclusion, p2_conclusion), dim=1)
            # x: (b, 2*d_model)
        else:
            x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
            # x: (b, 3*d_model)

        x = self.mlp_head(x)
        return x


# ==========================================================================
# Pre-configured BST variants — the single source of truth for flag combos.
# partial(BST, use_ppf=True, ...) creates a "pre-configured" version of BST
# that acts like a class — you can call BST_CG_AP(in_dim=72, ...) and the
# flags are already set. Same idea as functools.partial in any Python context.
# ==========================================================================
BST_0 = partial(BST, use_ppf=False, use_cg=False, use_ap=False)
BST_PPF = partial(BST, use_ppf=True, use_cg=False, use_ap=False)
BST_CG = partial(BST, use_ppf=True, use_cg=True, use_ap=False)
BST_AP = partial(BST, use_ppf=True, use_cg=False, use_ap=True)
BST_CG_AP = partial(BST, use_ppf=True, use_cg=True, use_ap=True)


if __name__ == '__main__':
    from pipeline.config import resolve_taxonomy
    n_class = resolve_taxonomy('une_v1_14').n_classes

    b, t, n = 1, 100, 2
    n_features = (17 + 19 * 1) * n
    pose = torch.randn((b, t, n, n_features), dtype=torch.float)
    shuttle = torch.randn((b, t, 2), dtype=torch.float)
    pos = torch.randn((b, t, n, 2), dtype=torch.float)
    videos_len = torch.tensor([t], dtype=torch.long).repeat(b)
    input_data = [pose, shuttle, pos, videos_len]

    # Test all variants produce valid output shapes
    variants = {
        'BST_0':     BST_0(in_dim=n_features, seq_len=t, n_class=n_class, d_model=100),
        'BST_PPF':   BST_PPF(in_dim=n_features, seq_len=t, n_class=n_class, d_model=100),
        'BST_CG':    BST_CG(in_dim=n_features, seq_len=t, n_class=n_class, d_model=100),
        'BST_AP':    BST_AP(in_dim=n_features, seq_len=t, n_class=n_class, d_model=100),
        'BST_CG_AP': BST_CG_AP(in_dim=n_features, seq_len=t, n_class=n_class, d_model=100),
    }
    for name, model in variants.items():
        output = model(*input_data)
        print(f"{name:10s} output shape: {output.shape}")

    # FLOP counting on BST_CG_AP
    from torch.utils.flop_counter import FlopCounterMode
    model = variants['BST_CG_AP']
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        output = model(*input_data)
    flops_per_forward = flop_counter.get_total_flops()
    print(f"\nFLOPs (per forward pass): {flops_per_forward / 1e9:.2f} GFLOPS")

    n_epochs_about = 350
    # on ShuttleSet
    n_training_samples = 25741
    n_validate_samples = 4241
    n_testing_samples = 3499

    training_flops = flops_per_forward * n_training_samples * n_epochs_about * 3
    validate_flops = flops_per_forward * n_validate_samples * n_epochs_about
    testing_flops = flops_per_forward * n_testing_samples
    print(f"Training FLOPs: {training_flops / 1e15:.2f} PFLOPs")
    print(f"Validating FLOPs: {validate_flops / 1e15:.2f} PFLOPs")
    print(f"Testing FLOPs (per 1000 instances): {flops_per_forward * 1000 / 1e12:.2f} TFLOPs")
    print(f"Testing FLOPs: {testing_flops / 1e12:.2f} TFLOPs")
    total_flops = training_flops + validate_flops + testing_flops
    print(f"Total FLOPs: {total_flops / 1e15:.2f} PFLOPs")
