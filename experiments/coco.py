import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from tqdm import trange
import matplotlib.pyplot as plt

from src.vit import ViT
from src.ctrl import StructuredDropout

epochs = 100
min_budget = 1
max_budget = 128

device = "cuda"

def eval(model):
  model = model.eval()

  cifar10 = tv.datasets.CIFAR10(
    "data", train=True, download=False,
    transform=tv.transforms.ToTensor(),
  )
  loader = torch.utils.data.DataLoader(cifar10, batch_size=333, shuffle=True)
  total = 0
  correct = 0
  for imgs, labels in loader:
    imgs = imgs.to(device)
    labels = labels.to(device)
    pred = model(imgs)

    loss = F.cross_entropy(pred, labels)
    pred_labels = F.softmax(pred,dim=-1).argmax(dim=-1)
    correct += (pred_labels == labels).sum().item()
    total += labels.shape[0]
  return correct/total

def main():
  cifar10 = tv.datasets.CIFAR10(
    "data", train=True, download=False,
    transform=tv.transforms.ToTensor(),
  )
  loader = torch.utils.data.DataLoader(cifar10, batch_size=333, shuffle=True)
  dropout = StructuredDropout(0.2, zero_pad=True, lower_bound=16)
  model = ViT(
    image_size=32,
    patch_size=8,
    num_classes=10,
    dim=128,
    depth=5,
    heads=4,
    mlp_dim=128,
    dropout = dropout,
    emb_dropout = dropout,
  ).to(device)

  opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

  t = trange(epochs)
  for i in t:
    for imgs, labels in loader:
      imgs = imgs.to(device)
      labels = labels.to(device)
      opt.zero_grad()
      pred = model(imgs)

      loss = F.cross_entropy(pred, labels)

      pred_labels = F.softmax(pred,dim=-1).argmax(dim=-1)
      correct = (pred_labels == labels).sum()
      t.set_postfix(
        L=f"{loss.item():.03f}",
        correct=f"{correct:03}/{labels.shape[0]:03}"
      )
      loss.backward()
      opt.step()
  accs = []
  budgets = range(0, 129)
  with torch.no_grad():
    for i in budgets:
      dropout.set_latent_budget(i)
      acc = eval(model)
      print(i, acc)
      accs.append(acc)
  print(accs)
  plt.plot(budgets, accs)
  plt.xlabel("budgets")
  plt.ylabel("accuracy")
  plt.title("latent budget relative to total accuracy")
  plt.savefig("budget.png")
