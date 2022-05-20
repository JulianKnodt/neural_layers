from .ctrl import StructuredDropout
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class DynConv2d(nn.Module):
  def __init__(
    self,
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups_list=[1],
    bias=True,
    padding_mode="zeros",
  ):
    super().__init__()
    self.conv = nn.Conv2d(
      in_channels,
      out_channels,
      kernel_size,
      stride,
      padding,
      dilation,
      groups, bias=True,
      padding_mode,
    )
    def forward(self, x, in_chans=None, out_chans=None):
      c = self.conv
      weight = c.weight
      bias = c.bias
      if in_chans is not None: weight = weight[:, :in_chans]
      if out_chans is not None:
        weight = weight[:out_chans]
        if bias is not None: bias = bias[:out_chans]
      return F.conv2d(x, weight, bias, c.stride, c.padding, c.dilation, c.groups)

class ConvSD(nn.Module):
  def __init__(self, conv, sd):
    super().__init__()
    self.conv = conv
    self.sd = sd
  def forward(self, x, in_chans=None, out_chans=None):
    # Do not have any designated output size
    if out_chans is None:
      out_chans, norm = None, 1
      if self.sd is not None: out_chans, norm = self.sd.pre_apply(self.conv.conv.out_channels)
    else:
      norm = self.conv.conv.out_channels/out_chans

    return F.relu(norm * self.conv(x, in_chans, out_chans), inplace=True), out_chans

