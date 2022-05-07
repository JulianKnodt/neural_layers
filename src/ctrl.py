import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import random

# Conditionally zeros out the last components of a vector
class StructuredDropout(nn.Module):
  def __init__(
    self,
    # chance of turning off bits.
    p=0.5,
    # the minimum number of features to always retain
    lower_bound:int=1,
    eval_size=None,
    zero_pad:bool = False,
    normalize:bool=True,

    step:int=1,
    dim=-1,
  ):
    assert(p >= 0)
    assert(p <= 1)
    assert(step >= 1)
    assert(lower_bound > 0), "Cannot use 0 or below as lower bound"

    super().__init__()
    self.p = p
    self.step = step
    self.lower_bound = lower_bound
    self.eval_size = eval_size
    self.normalize = normalize

    self.zero_pad = zero_pad
    self.dim = dim

  def inner(self, x):
    p = self.p
    upper = x.shape[-1]

    if not self.training:
      esz = self.eval_size
      if esz is None or esz > upper: return x

      cut = (upper/esz) * x[..., :esz]
      return cut if not self.zero_pad else F.pad(cut, (0,upper-esz))

    cutoff = self.cutoff(upper)
    if cutoff is None: return x

    cut = x[..., :cutoff]
    if self.normalize: cut = (upper/cutoff) * cut

    return cut if not self.zero_pad else F.pad(cut, (0, upper-cutoff))

  def forward(self, x): return self.inner(x.movedim(self.dim, -1)).movedim(-1, self.dim)

  def set_latent_budget(self, ls:int): self.eval_size = ls
  def cutoff(self, upper):
    if random.random() > self.p or self.lower_bound >= upper: return None
    return random.choice(range(self.lower_bound, upper, self.step))
  # Apply the linear layer that precedes x more cheaply.
  def pre_apply_linear(
    self,
    lin,
    x,
    output_features:int,
    input_features=None,
    bn = nn.Identity(),
  ):
    c = self.cutoff(output_features) if self.training else \
      (None if self.eval_size is None or self.eval_size >= output_features else self.eval_size)

    weight = lin.weight
    if input_features is not None:
      x = x[..., :input_features]
      weight = weight[:, :input_features]

    if c is None: return bn(F.linear(x, weight, lin.bias)), None

    bias = None if lin.bias is None else lin.bias[:c]
    cut = F.linear(x, weight[:c], bias)

    if isinstance(bn, nn.BatchNorm1d):
      bn_training = self.training or ((bn.running_mean is None) and (bn.running_var is None))
      cut = F.batch_norm(
        cut,
        bn.running_mean[:c] if not self.training or bn.track_running_stats else None,
        bn.running_var[:c] if not self.training or bn.track_running_stats else None,
        bn.weight[:c],
        bn.bias[:c],
        momentum=bn.momentum,
        eps=bn.eps,
        training=bn_training,
      )

    # This step is equivalent to applying the structured dropout
    # and it is important that it is applied after the batch norm
    cut = cut * (output_features/c)

    if self.zero_pad: cut = F.pad(cut, (0, output_features-c))
    return cut, c

class DynBatchNorm1d(nn.Module):
  def __init__(
    num_features,
    eps=1e-5,
    momentum=0.1,
    affine=True,
  ):
    self.bn = BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)
  def forward(self, x, cutoff):
    # TODO maybe this needs to take into account the number of input features?
    # If input features are missing, the expectation also goes down.
    # i.e. normalize by all_input_features/used_input_features
    c = cutoff
    bn = self.bn
    return F.batch_norm(
      x[:c],
      bn.running_mean[:c],
      bn.running_var[:c],
      weight=bn.weight[:c],
      bias=bn.bias[:c],
      training=self.training,
      momentum=bn.momentum,
      eps=bn.eps,
    )

# concatenates two vectors, but instead of
# stacking them on top of each other, interleave them.
def zip_concat(a, b, dim=-1):
  a = a.movedim(dim,-1)
  b = b.movedim(dim,-1)

  a_s = a.shape[-1]
  b_s = b.shape[-1]
  if a_s < b_s: a = F.pad(a, (0, b_s - a_s))
  elif a_s > b_s: b = F.pad(b, (0, a_s - b_s))

  return torch.stack([a, b],dim=-1).flatten(-2).movedim(-1, dim)

def index_opt(l, idx, default=None):
  try: return l[idx]
  except IndexError: return default

mlp_init_kinds = {
    None,
    "zero",
    "kaiming",
    "xavier",
    "siren",
}

class MLP(nn.Module):
  def __init__(
    self,
    in_features:int=571,
    out_features:int=3,
    hidden_sizes=[256] * 3,
    bias:bool=True,
    batch_norm:bool=False,
    skip=1000,

    activation=nn.LeakyReLU(inplace=True),
    init="xavier",
    dropout = StructuredDropout(p=0.5,lower_bound=32),
  ):
    assert init in mlp_init_kinds, f"Must use init kind, got {init} not in {mlp_init_kinds}"

    super().__init__()
    self.skip = skip

    skip_size = lambda i: hidden_sizes[0] if (i+1) % skip == 0 else 0

    self.layers = nn.ModuleList([
      nn.Linear(hidden_size + skip_size(i), hidden_sizes[i+1], bias=bias)
      for i, hidden_size in enumerate(hidden_sizes[:-1])
    ])

    if batch_norm:
      self.bns = nn.ModuleList([
        nn.BatchNorm1d(hs + skip_size(i)) for i, hs in enumerate(hidden_sizes)
      ])
    else:
      self.bns = nn.ModuleList([])

    assert(isinstance(dropout, StructuredDropout))
    self.dropout = dropout

    self.init = nn.Linear(in_features, hidden_sizes[0], bias=bias)
    self.out = nn.Linear(hidden_sizes[-1], out_features, bias=bias)

    self.act = activation

    if init is None: return

    weights = [self.init.weight, self.out.weight, *[l.weight for l in self.layers]]
    biases = [self.init.bias, self.out.bias, *[l.bias for l in self.layers]]

    if init == "zero":
      for t in weights: nn.init.zeros_(t)
      for t in biases: nn.init.zeros_(t)
    elif init == "siren":
      for t in weights:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(t)
        a = math.sqrt(6 / fan_in)
        nn.init._no_grad_uniform_(t, -a, a)
      for t in biases: nn.init.zeros_(t)
    elif init == "kaiming":
      for t in weights: nn.init.kaiming_normal_(t, mode="fan_out")
      for t in biases: nn.init.zeros_(t)

    self.ident = nn.Identity()

  def set_latent_budget(self,ls:int): self.dropout.set_latent_budget(ls)
  def number_inference_parameters(self, es:int=None):
    es = es or self.dropout.eval_size
    assert(es is not None)
    out = 0
    out += self.init.weight[:es, :].numel() + self.init.bias[:es].numel()
    for l in self.layers: out += l.weight[:es, :es].numel() + l.bias[:es].numel()
    out += self.init.weight[:, :es].numel() + self.init.bias.numel()
    return out
  def forward(self, p):
    flat = p.reshape(-1, p.shape[-1])

    init_x, init_cutoff = x, cutoff = self.dropout.pre_apply_linear(
      self.init, flat, self.init.out_features,
      bn=index_opt(self.bns, 0, self.ident),
    )

    for i, layer in enumerate(self.layers):
      x = self.act(x)

      # perform skip connections
      if (i+1) % self.skip == 0:
        x = zip_concat(init_x, x)
        # The sparsity is now 2x the least sparse prior vector.
        cutoff = None if init_cutoff is None or cutoff is None else (2 * max(init_cutoff, cutoff))

      x, cutoff = self.dropout.pre_apply_linear(
        layer, x, layer.out_features, cutoff,
        bn=index_opt(self.bns,i+1,self.ident),
      )

    x = self.act(x)
    return F.linear(x, self.out.weight[:, :x.shape[-1]], self.out.bias)\
      .reshape(*p.shape[:-1], self.out.out_features)
