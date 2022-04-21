import os
import random
import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

leaky_relu = nn.LeakyReLU(inplace=True)

class Tnet(nn.Module):
 def __init__(self, k=3, linear=nn.Linear, dropout=nn.Identity()):
    super().__init__()
    self.k=k
    self.convs = nn.Sequential(
      nn.Conv1d(k, 64, 1),
      nn.BatchNorm1d(64),
      leaky_relu,

      nn.Conv1d(64, 128, 1),
      nn.BatchNorm1d(128),
      leaky_relu,

      nn.Conv1d(128, 1024, 1),
      nn.BatchNorm1d(1024),
      leaky_relu,
    )

    self.fc = nn.Sequential(
      linear(1024,512),
      nn.BatchNorm1d(512),
      leaky_relu,
      dropout,

      linear(512,256),
      nn.BatchNorm1d(256),
      leaky_relu,
    )

    self.out = nn.Linear(256,k*k)


 def forward(self, input):
  x = self.fc(self.convs(input).max(dim=-1)[0])
  init = torch.eye(self.k, requires_grad=True, device=x.device)
  return self.out(x).reshape(-1,self.k,self.k) + init


class Transform(nn.Module):
  def __init__(self, dropout=nn.Identity()):
    super().__init__()
    self.input_tf = Tnet(k=3, dropout=dropout)
    self.input_to_feats = input_tf = nn.Sequential(
      nn.Conv1d(3,64,1),
      nn.BatchNorm1d(64),
      leaky_relu,
    )

    self.feature_transform = Tnet(k=64)
    self.feat_conv = nn.Sequential(
      nn.Conv1d(64,128,1),
      nn.BatchNorm1d(128),
      leaky_relu,

      nn.Conv1d(128,1024,1),
      nn.BatchNorm1d(1024),
    )


  def forward(self, x):
    matrix3x3 = self.input_tf(x)

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
    self.transform = Transform(dropout=dropout)

    self.compress = nn.Sequential(
      linear(1024, 512),
      nn.BatchNorm1d(512),
      leaky_relu,
      dropout,

      linear(512, 256),
      nn.BatchNorm1d(256),
      leaky_relu,
      dropout,

      nn.Linear(256, classes),
    )

  def forward(self, input):
    x, matrix3x3, matrix64x64 = self.transform(input)
    return self.compress(x), matrix3x3, matrix64x64

# reads just the points from an off file
def read_off(f):
  with open(f, "r") as f:
    first = f.readline().lower().strip()
    assert(first == "off"), first
    v, faces, e = [int(v) for v in f.readline().split()]
    lines = f.readlines()
    verts = torch.tensor([
      [float(val) for val in line.split()]
      for line in lines[:v]
    ])
    faces = torch.tensor([
      [int(val) for val in line.split()]
      for line in lines[v:v+faces]
    ])
  return verts, faces


class ModelNet10(torch.utils.data.Dataset):
  '''Object classification on ModelNet'''
  def __init__(self, root, npoints=1000, train=True):
    self.root = root

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
    self.train = train
    for item in os.listdir(self.root):
      if not os.path.isdir(os.path.join(self.root, item)): continue
      for f in os.listdir(os.path.join(self.root, item, FLAG)):
        if not f.endswith('.off'): continue
        self.datapath.append((os.path.join(self.root, item, FLAG, f), self.cat[item]))
    self.npoints = npoints

  def __getitem__(self, idx):
    fn = self.datapath[idx]
    points, faces = read_off(fn[0])
    replace = points.shape[0] < self.npoints
    choice = np.random.choice(points.shape[0], self.npoints, replace=replace)
    points = points[choice, :]

    label = fn[1]

    if self.train:
      points = torch.randn_like(points) * 1e-1 + points
      theta = random.random() * 2. * math.pi
      rot_matrix = torch.tensor([[ math.cos(theta), -math.sin(theta),    0],
                                 [ math.sin(theta),  math.cos(theta),    0],
                                 [0,                             0,      1]])
      points = torch.matmul(rot_matrix[None], points[...,None]).squeeze(-1)


    points = points - points.mean(dim=0)
    points = points/points.norm(dim=1).max(dim=0)[0]

    label = torch.tensor([label], dtype=torch.int64)
    return points, label

  def __len__(self): return len(self.datapath)

