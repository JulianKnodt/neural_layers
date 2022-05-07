import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from tqdm import trange, tqdm

from src.ctrl import StructuredDropout
from src.pointnet import ModelNet10, PointNet
from .utils import plot_budgets, plot_number_parameters

epochs = 100
max_budget = 512
step = 1
p = 0.5

device = "cuda"

def eval(model, latent:int, budgeter):
  if latent is not None: budgeter.set_latent_budget(latent)
  modelnet10 = ModelNet10("data/ModelNet10", train=False)
  loader = torch.utils.data.DataLoader(modelnet10, batch_size=64, shuffle=False)
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
  model = PointNet(dropout=StructuredDropout(p, step=step)).to(device)
  #model = torch.load(f"models/pointnet{p}.pt")
  sd = model.dropout

  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  alpha = 1e-3
  t = trange(epochs)
  for i in t:
    torch.save(model, f"models/pointnet{p}.pt")
    for pts, label in loader:
      opt.zero_grad(set_to_none=True)

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
      )
  model = model.eval()

  budgets = range(1, max_budget+1)
  params = [model.num_parameters(es) for es in budgets]
  plot_number_parameters(budgets, params, "FC Layer Parameters", title="PointNet")

  with torch.no_grad():
    #print("No latent budget", eval(model, None, None))
    budgets = range(1, max_budget+1, step)
    accs = [eval(model, i, sd) for i in tqdm(budgets)]
  plot_budgets(budgets, accs)

if __name__ == "__main__": main()
