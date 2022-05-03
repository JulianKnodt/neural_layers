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

    self.zero_pad = zero_pad
    self.dim = dim

  def forward(self, x):
    p = self.p
    upper = x.shape[-1]

    if not self.training:
      esz = self.eval_size
      if esz is None or esz >= upper: return x

      cut = (upper/esz) * x[..., :esz]
      return cut if not self.zero_pad else F.pad(cut, (0,upper-esz))

    cutoff = self.cutoff(upper)
    if cutoff is None: return x

    cut = (upper/cutoff) * x[..., :cutoff]
    return cut if not self.zero_pad else F.pad(cut, (0, upper-cutoff))

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
  ):
    cutoff = self.cutoff(output_features) if self.training else self.eval_size

    weight = lin.weight
    if input_features is not None:
      x = x[..., :input_features]
      weight = weight[:, :input_features]

    if cutoff is None: return F.linear(x, weight, lin.bias), None


    bias = None if lin.bias is None else lin.bias[:cutoff]
    norm = output_features/cutoff
    cut = F.linear(x, weight[:cutoff], bias) * norm
    if self.zero_pad: cut = F.pad(cut, (0, output_features-cutoff))
    return cut, cutoff

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
    # instead of outputting a single color, output multiple colors
    bias:bool=True,
    skip=1000,

    activation=nn.LeakyReLU(inplace=True),
    init="xavier",
    dropout = StructuredDropout(p=0.2,lower_bound=3, step=5),
  ):
    assert init in mlp_init_kinds, f"Must use init kind, got {init} not in {mlp_init_kinds}"

    super().__init__()
    self.skip = skip

    self.layers = nn.ModuleList([
      nn.Linear(hidden_size, hidden_sizes[i+1], bias=bias)
      for i, hidden_size in enumerate(hidden_sizes[:-1])
    ])

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
  def number_inference_parameters(self):
    es = self.dropout.eval_size
    assert(es is not None)
    out = 0
    out += self.init.weight[:es, :].numel() + self.init.bias[:es].numel()
    for l in self.layers: out += l.weight[:es, :es].numel() + l.bias[:es].numel()
    out += self.init.weight[:, :es].numel() + self.init.bias.numel()
    return out
  def forward(self, p):
    flat = p.reshape(-1, p.shape[-1])

    x, cutoff = self.dropout.pre_apply_linear(
      self.init, flat, self.init.out_features,
    )
    init_x = x
    init_cutoff = cutoff

    for i, layer in enumerate(self.layers):
      # perform skip connections
      if i != 0 and i % self.skip == 0:
        if init_cutoff > cutoff:
          x, skip = init_x, x
          cutoff = init_cutoff
        else: skip = init_x
        x[..., :c] = x[..., :c] + init_x[..., :c]

      x = self.act(x)
      x, cutoff = self.dropout.pre_apply_linear(layer, x, layer.out_features, cutoff)

    out_size = self.out.out_features

    return F.linear(x, self.out.weight[:, :x.shape[-1]], self.out.bias)\
      .reshape(*p.shape[:-1], out_size)
