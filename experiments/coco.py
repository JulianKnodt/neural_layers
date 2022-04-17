import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

epochs = 80
min_budget = 1
max_budget = 128

device = "cuda"

def main():
  coco = tv.datasets.CocoDetection(
    "data/coco/train2014",
    "data/coco/annotations/instances_train2014.json",
    transform=tv.transforms.ToTensor()
  )
  loader = torch.utils.data.DataLoader(coco, batch_size=100, shuffle=True)
  model = ...

  opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0)
  t = trange(epochs)
  for i in t:
    for imgs, labels in loader:
      print(imgs.shape, labels.shape)
      exit()
      opt.zero_grad()
      pred = model(imgs)

      loss = F.cross_entropy(pred, label)
      t.set_postfix(L=f"{loss.item():.03f}")
      loss.backward()
      opt.step()
