import torch
import numpy as np
from sklearn.model_selection import KFold
from model import *
from data import *
from trainer import *
from sampler import *

X = np.zeros((1000, 2, 3))
for i in range(1000):
    X[i, 1, 2] = 1
Z = np.ones((1000, 2, 1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = Diffusion()

train_dataset = CustomDataset(X, Z)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
val_dataset = CustomDataset(X, Z)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=True, drop_last=True)

trainer = Trainer(model, lr=1.0e-3, epoch=100, device=device)
res = trainer.train(train_dataloader, val_dataloader)

sampler = Sampler(model, device=device)
sampler.sample(n_step=10000)
