import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import random

mlp_init_kinds = {
    None,
    "zero",
    "kaiming",
    "xavier",
    "siren",
}

# An upper triangular linear layer.
# This retains sparsity in further layers,
# because we know the output structure given a sparse input.
class TriangleLinear(nn.Module):
  def __init__(
    self,
    in_features:int,
    out_features:int,
    bias:bool=True,
    flip:bool=True,
    backflow:int=4,
  ):
    super().__init__()
    assert(backflow >= 0), "must pass positive backflow"
    self.diagonal = min(in_features - out_features, 0) - backflow
    self.backflow=backflow

    self.register_buffer("mask", torch.triu(
      torch.ones(out_features, in_features, dtype=torch.bool),
      diagonal=self.diagonal,
    ))
    self.out_features = out_features
    self.in_features = in_features
    self.out_features = out_features
    self.flip = flip

    self.weight = nn.Parameter(torch.rand(self.mask.sum()))
    if bias: self.bias = nn.Parameter(torch.rand(out_features))
    else: self.register_parameter("bias", None)
  def get_matrix(self, sz=None):
    if sz is None:
      return torch.zeros_like(self.mask, dtype=torch.float).masked_scatter(mask, self.weight)
    assert(sz <= self.in_features),\
      f"Must pass less parameters to triangle layer {layer.shape} {sz}"
    mask = self.mask[:sz - self.diagonal, :sz]
    return torch.zeros_like(mask, dtype=torch.float)\
      .masked_scatter(mask, self.weight[:mask.sum()])

  def forward(self, x):
    layer = self.get_matrix(x.shape[-1])

    out = torch.sum(layer * x[..., None, :], dim=-1)
    if self.bias is not None: out = out + self.bias[:out.shape[-1]]
    return out.flip([-1]) if self.flip else out

  # Messing around with using solving instead of using matrix multiply
  def solve_forward(self, b):
    A = self.get_matrix(b.shape[-1])
    assert(A.shape[0] == A.shape[1])
    X = torch.linalg.triangular_solve(A, b)
    if self.bias is not None: X = X - self.bias[:out.shape[-1]]
    return X

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
  def pre_apply_linear(self, lin, x, output_features:int, input_features=None):
    cutoff = self.cutoff(output_features) if self.training else self.eval_size


    weight = lin.weight
    if input_features is not None:
      x = x[..., :input_features]
      weight = weight[:, :input_features]

    if cutoff is None: return F.linear(x, weight, lin.bias), None


    bias = None if lin.bias is None else lin.bias[:cutoff]
    cut = F.linear(x, weight[:cutoff], bias) * output_features/cutoff
    return (cut if not self.zero_pad else F.pad(cut, (0, output_features-cutoff))), cutoff


class TriangleMLP(nn.Module):
  "MLP which uses triangular layers for efficiency"
  def __init__(
    self,
    in_features:int,
    out_features:int=3,
    hidden_sizes=[256] * 3,
    # instead of outputting a single color, output multiple colors
    skip:int=3,
    bias:bool=True,
    backflow:int=0,
    flip=True,

    activation=nn.LeakyReLU(inplace=True),
    init=None,
    init_dropout = StructuredDropout(p=0.5,lower_bound=1),
  ):
    assert init in mlp_init_kinds, "Must use init kind"
    super(TriangleMLP, self).__init__()

    self.in_features = in_features

    self.skip = skip

    hidden_layers = [
        TriangleLinear(
            hidden_size, hidden_sizes[i+1],
            bias=bias, backflow=backflow, flip=flip and (i%2 == 1),
        )
        for i, hidden_size in enumerate(hidden_sizes[:-1])
    ]

    self.init = nn.Linear(in_features, hidden_sizes[0], bias=bias)
    assert(isinstance(init_dropout, StructuredDropout))
    self.init_dropout = init_dropout

    self.layers = nn.ModuleList(hidden_layers)

    self.out = nn.Linear(hidden_sizes[-1], out_features, bias=bias)

    self.activation = activation

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

  def set_latent_budget(self,ls:int): self.init_dropout.set_latent_budget(ls)
  def forward(self, p):
    flat = p.reshape(-1, p.shape[-1])

    #init = x = self.init_dropout(self.init(flat))
    init = x = self.init_dropout.pre_apply_linear(self.init, flat, self.init.out_features)

    for i, layer in enumerate(self.layers):
      if i != 0 and i != len(self.layers) - 1 and (i % self.skip) == 0:
        # TODO maybe experiment with concatenating the original vector on top here?
        #x = x + F.pad(init, (0, x.shape[-1]-init.shape[-1]))
        ...
      x = layer(self.activation(x))

    out_size = self.out.out_features

    x = self.activation(x)
    return F.linear(x, self.out.weight[:, :x.shape[-1]], self.out.bias)\
      .reshape(p.shape[:-1] + (out_size,))

class MLP(nn.Module):
  def __init__(
    self,
    in_features:int=571,
    out_features:int=3,
    hidden_sizes=[256] * 3,
    # instead of outputting a single color, output multiple colors
    bias:bool=True,

    activation=nn.LeakyReLU(inplace=True),
    init="xavier",
    dropout = StructuredDropout(p=0.5,lower_bound=3),
  ):
    assert init in mlp_init_kinds, f"Must use init kind, got {init} not in {mlp_init_kinds}"

    super().__init__()

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

    x, cutoff = self.dropout.pre_apply_linear(self.init, flat, self.init.out_features)

    for i, layer in enumerate(self.layers):
      x = self.act(x)
      x, cutoff = self.dropout.pre_apply_linear(layer, x, layer.out_features, cutoff)

    out_size = self.out.out_features

    return F.linear(x, self.out.weight[:, :x.shape[-1]], self.out.bias)\
      .reshape(*p.shape[:-1], out_size)
