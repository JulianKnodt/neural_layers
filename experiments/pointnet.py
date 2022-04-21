import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from tqdm import trange, tqdm

from src.ctrl import TriangleMLP, StructuredDropout, TriangleLinear
from src.pointnet import ModelNet10, PointNet
from .utils import plot_budgets

epochs = 50
min_budget = 1
max_budget = 512
step = 4

device = "cuda"

def eval(model, latent:int, budgeter):
  if latent is not None: budgeter.set_latent_budget(latent)
  modelnet10 = ModelNet10("data/ModelNet10", train=False)
  loader = torch.utils.data.DataLoader(modelnet10, batch_size=50, shuffle=False)
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
  loader = torch.utils.data.DataLoader(modelnet10, batch_size=32, shuffle=True)
  sd = StructuredDropout(0.6, zero_pad=True,step=step)
  model = PointNet(dropout=sd).to(device)
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  alpha = 1e-3
  t = trange(epochs)
  for i in t:
    for pts, label in loader:
      opt.zero_grad()

      pts = pts.to(device).transpose(-1,-2)
      label = label.to(device).squeeze(-1)
      pred,m3,m64 = model(pts)

      loss = F.cross_entropy(pred, label)
      B = pts.shape[0]
      loss = loss + alpha * \
        torch.linalg.norm(torch.eye(3, device=device) - torch.bmm(m3,m3.transpose(1,2)))/B
      loss = loss + alpha * \
        torch.linalg.norm(torch.eye(64, device=device) - torch.bmm(m64,m64.transpose(1,2)))/B

      loss.backward()
      opt.step()

      pred_labels = F.softmax(pred,dim=-1).argmax(dim=-1)
      correct = (pred_labels == label).sum()
      t.set_postfix(
        L=f"{loss.item():.03f}", correct=f"{correct:03}/{label.shape[0]:03}",
        #lr=f"{sched.get_last_lr()[0]:.01e}"
      )

  model = model.eval()
  with torch.no_grad():
    print("No latent budget", eval(model, None, None))
    budgets = range(1, max_budget+1, step)
    accs = [eval(model, i, sd) for i in tqdm(budgets)]
    print(accs)

  plot_budgets(budgets, accs)

if __name__ == "__main__": main()
