import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from tqdm import trange

from src.ctrl import TriangleMLP, StructuredDropout, TriangleLinear

epochs = 80
min_budget = 1
max_budget = 128

device = "cuda"

def eval(model, latent:int):
  if latent is not None: model.set_latent_budget(latent)
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
  model = TriangleMLP(
    in_features=28 * 28,
    out_features=10,
    init_dropout=StructuredDropout(lower_bound=min_budget,p=0.9),
    hidden_sizes=[max_budget]*2,
    backflow=5,
    skip=1000,
    flip=True,
  ).to(device)
  rounded_budget = math.floor(math.sqrt(2) * max_budget)
  model = nn.Sequential(
    nn.Linear(28 * 28, rounded_budget),
    #TriangleLinear(28 * 28, max_budget, backflow=10,flip=True),
    nn.LeakyReLU(),
    #nn.Linear(rounded_budget, rounded_budget),
    TriangleLinear(rounded_budget, rounded_budget, backflow=10,flip=True),
    nn.LeakyReLU(),
    #nn.Dropout(),
    #StructuredDropout(zero_pad=True),
    nn.Linear(rounded_budget, 10),
  ).to(device)
  opt = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=0)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50_000)
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
      sched.step()

      pred_labels = F.softmax(pred,dim=-1).argmax(dim=-1)
      correct = (pred_labels == label).sum()
      t.set_postfix(
        L=f"{loss.item():.03f}", correct=f"{correct:03}/{label.shape[0]:03}",
        lr=f"{sched.get_last_lr()[0]:.01e}"
      )
  if isinstance(model, TriangleMLP):
    with torch.no_grad():
      accuracies = [eval(model, i) for i in range(1, max_budget+1)]
      print(accuracies)
  else:
    print(eval(model, None))

if __name__ == "__main__": main()
