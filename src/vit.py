# Modified version of:
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ctrl import TriangleMLP, StructuredDropout, TriangleLinear

class PreNorm(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.fn = fn
  def forward(self, x, **kwargs): return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
  def __init__(
    self,
    dim,
    heads:int = 8,
    dim_head:int = 64,
    dropout=nn.Dropout(0.),
  ):
    super().__init__()
    inner_dim = dim_head *  heads
    project_out = not (heads == 1 and dim_head == dim)

    self.heads = heads
    self.scale = dim_head ** -0.5

    self.dropout = dropout

    self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

    self.to_out = nn.Identity if not project_out else nn.Sequential(
      nn.Linear(inner_dim, dim),
      dropout,
    )

  def forward(self, x):
    qkv = self.to_qkv(x).chunk(3, dim = -1)
    q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1,2), qkv)
    print(q.shape, k.shape)
    exit()

    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

    attn = F.softmax(dots, dim=-1)
    attn = self.dropout(attn)

    out = torch.matmul(attn, v)\
      .transpose(1,2)\
      .flatten(2)
    return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim:int, depth:int, heads:int, dim_head:int, mlp_dim:int, dropout:float = 0.):
        super().__init__()
        self.layers = nn.ModuleList([
          nn.ModuleList([
            PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
          ])
          for _ in range(depth)
        ])
    def forward(self, x):
      for attn, ff in self.layers:
        x = attn(x) + x
        x = ff(x) + x
      return x

