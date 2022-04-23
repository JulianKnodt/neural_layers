import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from tqdm import trange, tqdm

from src.ctrl import StructuredDropout, MLP
from .utils import plot_budgets, plot_timing

epochs = 50
min_budget = 1
max_budget = 256

device = "cuda"

def eval(model, latent:int, budgeter):
  if latent is not None: budgeter.set_latent_budget(latent)
  model = model.eval()
  mnist = tv.datasets.MNIST("data", train=False, download=True, transform=tv.transforms.ToTensor())
  loader = torch.utils.data.DataLoader(mnist, batch_size=500, shuffle=False)
  total = 0
  correct = 0
  for img, label in loader:
    img = img.flatten(1).to(device)
    label = label.to(device)
    pred = model(img)
    pred_labels = F.softmax(pred,dim=-1).argmax(dim=-1)
    correct += (pred_labels == label).sum().item()
    total += label.shape[0]
  #print(f"latent({latent:03}): correct={correct}/{total}")
  return correct/total

def main():
  mnist = tv.datasets.MNIST("data", download=True, transform=tv.transforms.ToTensor())
  loader = torch.utils.data.DataLoader(mnist, batch_size=500, shuffle=True)
  # in order to test without structured Dropout, set p to 0.
  sd = StructuredDropout(0.8, zero_pad=True)
  model = MLP(
    in_features=28 * 28, out_features=10,
    hidden_size=[256] * 2,
    bias=True,
    dropout=sd,
  )
  opt = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=0)
  #sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50_000)
  t = trange(epochs)
  for i in t:
    for img, label in loader:
      opt.zero_grad()
      img = img.flatten(1).to(device)
      label = label.to(device)
      pred = model(img)
      loss = F.cross_entropy(pred, label)
      loss.backward()
      opt.step()
      #sched.step()

      pred_labels = F.softmax(pred,dim=-1).argmax(dim=-1)
      correct = (pred_labels == label).sum()
      t.set_postfix(
        L=f"{loss.item():.03f}", correct=f"{correct:03}/{label.shape[0]:03}",
        #lr=f"{sched.get_last_lr()[0]:.01e}"
      )
  budgets = range(1, max_budget+1)
  with torch.no_grad():
    accs = []
    times = []
    for b in tqdm(budgets):
      start = time.time()
      acc = eval(model, i, sd)
      times.append(time.time() - start)
    print(accs)
    print(times)
  plot_budgets(budgets, accs)
  plot_timing(budgets, times)

if __name__ == "__main__": main()
