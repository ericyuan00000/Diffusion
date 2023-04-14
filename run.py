import torch
import numpy as np
from sklearn.model_selection import KFold
from model import *
from data import *
from trainer import *
from sampler import *

X = np.zeros((10000, 2, 3))
for i in range(10000):
    X[i, 1, 2] = 1
Z = np.ones((10000, 2, 1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Diffusion(n_layer=9, n_feat=256, n_atomtype=2)

train_dataset = CustomDataset(X, Z)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
val_dataset = CustomDataset(X, Z)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=True, drop_last=True)

trainer = Trainer(model, lr=1.0e-3, epoch=1000, device=device)
res = trainer.train(train_dataloader, val_dataloader)

sampler = Sampler(model, device=device)
sampler.sample(n_sample=100, n_step=10000)
