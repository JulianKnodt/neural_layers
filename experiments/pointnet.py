import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from tqdm import trange

from src.ctrl import TriangleMLP, StructuredDropout, TriangleLinear
from src.pointnet import ModelNet10, PointNet

epochs = 50
min_budget = 1
max_budget = 128

device = "cuda"

def eval(model, latent:int, budgeter):
  if latent is not None: budgeter.set_latent_budget(latent)
  model = model.eval()
  mnist = ModelNet10("data/ModelNet10", train=False)
  loader = torch.utils.data.DataLoader(mnist, batch_size=50, shuffle=False)
  total = 0
  correct = 0
  for pts, label in loader:
    pts = pts.to(device).transpose(-1,-2)
    label = label.to(device).squeeze(-1)
    pred,_,_ = model(pts)

    pred_labels = F.softmax(pred,dim=-1).argmax(dim=-1)
    correct += (pred_labels == label).sum().item()
    total += label.shape[0]

  return correct/total

def main():
  modelnet10 = ModelNet10("data/ModelNet10", train=True)
  loader = torch.utils.data.DataLoader(modelnet10, batch_size=50, shuffle=True)
  sd = StructuredDropout(0.8, zero_pad=True)
  model = PointNet(linear=TriangleLinear).to(device)
  opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50_000)
  t = trange(epochs)
  for i in t:
    for pts, label in loader:
      opt.zero_grad()
      pts = pts.to(device).transpose(-1,-2)
      label = label.to(device).squeeze(-1)
      pred,_,_ = model(pts)
      loss = F.cross_entropy(pred, label)
      loss.backward()
      opt.step()
      sched.step()

      pred_labels = F.softmax(pred,dim=-1).argmax(dim=-1)
      correct = (pred_labels == label).sum()
      t.set_postfix(
        L=f"{loss.item():.03f}", correct=f"{correct:03}/{label.shape[0]:03}",
        lr=f"{sched.get_last_lr()[0]:.01e}"
      )
  with torch.no_grad():
    accuracies = [eval(model, i, sd) for i in range(1, max_budget+1)]
    print(accuracies)

if __name__ == "__main__": main()
