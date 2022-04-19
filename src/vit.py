# Modified version of:
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
from einops.layers.torch import Rearrange

from src.ctrl import TriangleMLP, StructuredDropout, TriangleLinear

class PreNorm(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.fn = fn
  def forward(self, x, **kwargs): return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, dropout = nn.Dropout(0.)):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(dim, hidden_dim),
      nn.GELU(),
      dropout,
      nn.Linear(hidden_dim, dim),
      dropout,
    )
  def forward(self, x): return self.net(x)


class Attention(nn.Module):
  def __init__(
    self,
    dim,
    heads:int = 8,
    dim_head:int = 64,
    dropout=nn.Dropout(0.),
  ):
    super().__init__()
    inner_dim = dim_head * heads
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

def pair(v): return v if isinstance(v, tuple) else (v, v)

class ViT(nn.Module):
  def __init__(
    self, *, image_size, patch_size,
    num_classes, dim, depth, heads, mlp_dim,
    pool = 'cls', channels = 3, dim_head = 64,
    dropout = nn.Dropout(0.),
    emb_dropout = nn.Dropout(0.),
  ):
    super().__init__()
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)

    assert image_height % patch_height == 0 and image_width % patch_width == 0, \
      'Image dimensions must be divisible by the patch size.'
    num_high = image_height // patch_height
    num_wide = image_width // patch_width

    num_patches = num_high * num_wide

    patch_dim = channels * patch_height * patch_width
    assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

    self.to_patch_embedding = nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        nn.Linear(patch_dim, dim),
    )

    self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
    self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
    self.dropout = emb_dropout

    self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    self.pool = pool
    self.to_latent = nn.Identity()

    self.mlp_head = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes)
    )

    self.from_patch_embedding = nn.Sequential(
      nn.Linear(dim, patch_dim),
      Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', w=num_wide, h=num_high, p1 = patch_height, p2 = patch_width),
    )
  def forward(self, img):
    x = self.to_patch_embedding(img)
    b, n, _ = x.shape

    cls_tokens = self.cls_token.expand(b, -1, -1)
    x = torch.cat([cls_tokens, x], dim=1)
    x += self.pos_embedding[:, :(n + 1)]
    x = self.dropout(x)

    x = self.transformer(x)

    # sequence of outputs:
    x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

    return self.mlp_head(self.to_latent(x))
