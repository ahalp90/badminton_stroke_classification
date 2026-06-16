# TemPose: a new skeleton-based transformer model designed for fine-grained motion recognition in badminton
# (2023/08) https://ieeexplore.ieee.org/document/10208321
# Authors: Magnus Ibh, Stella Grasshof, Dan Witzner, Pascal Madeleine

# Modified by Jing-Yuan Chang

import torch
from torch import nn, Tensor


class MLP(nn.Module):
    '''Same as MLP_Block in TemPose paper.'''
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hd_dim),
            nn.GELU(),
            nn.Dropout(drop_p, inplace=True),
            nn.Linear(hd_dim, out_dim)
        )

    def forward(self, x: Tensor):
        return self.mlp(x)


class MLP_Head(nn.Module):
    '''Same as MLP_Head in TemPose.'''
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, out_dim, hd_dim, drop_p)

    def forward(self, x: Tensor):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class FeedForward(nn.Module):
    '''Same as FeedForward in TemPose.'''
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, hd_dim, drop_p)
        self.dropout = nn.Dropout(drop_p, inplace=True)

    def forward(self, x: Tensor):
        x = self.mlp(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    '''Same as Attention in TemPose.'''
    def __init__(self, d_model, d_head, n_head, drop_p) -> None:
        super().__init__()
        d_cat = d_head * n_head

        self.h = n_head
        self.to_qkv = nn.Linear(d_model, d_cat * 3, bias=False)
        self.scale = d_head**-0.5

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(drop_p)  # This shouldn't be inplace.
        )
        
        self.tail = nn.Sequential(
            nn.Linear(d_cat, d_model),
            nn.Dropout(drop_p, inplace=True)
        ) if n_head != 1 or d_cat != d_model else nn.Identity()

    def forward(self, x: Tensor, mask: Tensor = None):
        # x: (b*n, t, d_model)
        bn, t, _ = x.shape

        qkv: Tensor = self.to_qkv(x)
        qkv = qkv.view(bn, t, self.h, -1).chunk(3, dim=-1)
        q, k, v = map(lambda ts: ts.transpose(1, 2), qkv)
        # q, k, v: (bn, h, t, d_head)

        dots: Tensor = (q.contiguous() @ k.transpose(-1, -2).contiguous()) * self.scale
        # dots: (bn, h, t, t)
        if mask is not None:
            # mask: (bn, t)
            mask = mask.view(bn, 1, 1, t)
            dots = dots.masked_fill(~mask, -torch.inf)
        
        coef = self.attend(dots)
        attention: Tensor = coef @ v.contiguous()
        # attention: (bn, h, t, d_head)
        
        out = attention.transpose(1, 2).reshape(bn, t, -1)
        # out: (bn, t, h*d_head)
        out = self.tail(out)
        return out  # (bn, t, d_model)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_head, n_head, hd_mlp, drop_p) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, d_head, n_head, drop_p)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_model, hd_mlp, drop_p)

    def forward(self, x: Tensor, mask=None):
        z = self.layer_norm1(x)
        x = self.attn(z, mask) + x
        z = self.layer_norm2(x)
        x = self.ff(z) + x
        return x


class TransformerEncoder(nn.Module):
    '''Same as Transformer in TemPose.'''
    def __init__(self, d_model, d_head, n_head, depth, hd_mlp, drop_p) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, d_head, n_head, hd_mlp, drop_p)
             for _ in range(depth)]
        )

    def forward(self, x: Tensor, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TCN(nn.Module):
    '''Same as TCN in TemPose. There is a bit different from the original TCN.'''
    def __init__(self, in_channel, channels: list[int], kernel_size=5, drop_p=0.3) -> None:
        '''`kernel_size` should be an odd number, so the output sequence length can remain the same as input.'''
        super().__init__()
        layers = []
        for i in range(len(channels)):
            in_ch = in_channel if i == 0 else channels[i-1]
            out_ch = channels[i]
            
            dilation = i * 2 + 1
            padding = (kernel_size - 1) * dilation // 2
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(drop_p, inplace=True)
            ]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: Tensor):
        return self.net(x)
