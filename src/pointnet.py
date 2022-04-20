import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Tnet(nn.Module):
 def __init__(self, k=3, linear=nn.Linear):
    super().__init__()
    self.k=k
    self.convs = nn.Sequential(
      nn.Conv1d(k,64,1),
      nn.BatchNorm1d(64),
      nn.LeakyReLU(),

      nn.Conv1d(64,128,1),
      nn.BatchNorm1d(128),
      nn.LeakyReLU(),

      nn.Conv1d(128,1024,1),
      nn.BatchNorm1d(1024),
      nn.LeakyReLU(),
    )

    self.fc = nn.Sequential(
      linear(1024,512),
      nn.BatchNorm1d(512),
      nn.LeakyReLU(),

      linear(512,256),
      nn.BatchNorm1d(256),
      nn.LeakyReLU(),
    )

    self.out = nn.Linear(256,k*k)


 def forward(self, input):
  # input.shape == (bs,n,3)
  bs = input.size(0)
  x = self.fc(self.convs(input).max(dim=-1)[0])

  #initialize as identity
  init = torch.eye(self.k, requires_grad=True, device=x.device)
  return self.out(x).reshape(-1,self.k,self.k) + init


class Transform(nn.Module):
  def __init__(self):
    super().__init__()
    self.input_tf = Tnet(k=3)
    self.input_to_feats = input_tf = nn.Sequential(
      nn.Conv1d(3,64,1),
      nn.BatchNorm1d(64),
      nn.LeakyReLU(),
    )

    self.feature_transform = Tnet(k=64)
    self.feat_conv = nn.Sequential(
      nn.Conv1d(64,128,1),
      nn.BatchNorm1d(128),
      nn.LeakyReLU(),

      nn.Conv1d(128,1024,1),
      nn.BatchNorm1d(1024),
    )


  def forward(self, x):
    matrix3x3 = self.input_tf(x)
    # batch matrix multiplication
    xb = torch.bmm(x.transpose(1,2), matrix3x3).transpose(1,2)
    xb = self.input_to_feats(xb)

    matrix64x64 = self.feature_transform(xb)
    xb = torch.bmm(xb.transpose(1,2), matrix64x64).transpose(1,2)
    xb = self.feat_conv(xb)
    output = xb.max(dim=-1)[0]
    return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
  def __init__(self, classes = 10, dropout=nn.Dropout(0.3), linear=nn.Linear):
    super().__init__()
    self.transform = Transform()

    self.compress = nn.Sequential(
      linear(1024, 512),
      nn.BatchNorm1d(512),
      nn.LeakyReLU(),

      linear(512, 256),
      dropout,
      nn.BatchNorm1d(256),
      nn.LeakyReLU(),

      nn.Linear(256, classes),
    )

  def forward(self, input):
    x, matrix3x3, matrix64x64 = self.transform(input)
    return self.compress(x), matrix3x3, matrix64x64

def scale_linear_bycolumn(rawdata, high=1.0, low=0.0):
    mins = rawdata.min(dim=0)[0]
    maxs = rawdata.max(dim=0)[0]
    rng = maxs - mins
    return high - (high-low)*(maxs-rawdata)/(rng+np.finfo(np.float32).eps)

# reads just the points from an off file
def off_reader(f):
  with open(f, "r") as f:
    first = f.readline().lower().strip()
    assert(first == "off"), first
    v, faces, e = [int(v) for v in f.readline().split()]
    out = torch.tensor([
      [float(val) for val in line.split()]
      for line in f.readlines(v)
    ])
  return out


class ModelNet10(torch.utils.data.Dataset):
  '''Object classification on ModelNet'''
  def __init__(self, root, npoints=1024, train=True):
    self.root = root
    self.npoints = npoints
    self.cat = {
      "bathtub": 0,
      "bed": 1,
      "chair": 2,
      "desk": 3,
      "dresser": 4,
      "monitor": 5,
      "night_stand": 6,
      "sofa": 7,
      "table": 8,
      "toilet": 9,
    }

    self.num_classes = len(self.cat)
    self.datapath = []
    FLAG = 'train' if train else 'test'
    for item in os.listdir(self.root):
      if not os.path.isdir(os.path.join(self.root, item)): continue
      for f in os.listdir(os.path.join(self.root, item, FLAG)):
        if not f.endswith('.off'): continue
        self.datapath.append((os.path.join(self.root, item, FLAG, f), self.cat[item]))


  def __getitem__(self, idx):
    fn = self.datapath[idx]
    points = off_reader(fn[0])
    label = fn[1]
    replace = points.shape[0] < self.npoints
    choice = np.random.choice(points.shape[0], self.npoints, replace=replace)
    points = points[choice, :]
    points = scale_linear_bycolumn(points)
    label = torch.tensor([label], dtype=torch.int)
    return points, label

  def __len__(self): return len(self.datapath)

