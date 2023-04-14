import torch
import numpy as np
from sklearn.model_selection import KFold
from model import *
from data import *
from trainer import *

X = np.zeros((1000, 2, 3))
for i in range(1000):
    X[i, 1, 2] = 1
Z = np.ones((1000, 2, 1))

res = []
for train_selector, val_selector in KFold(n_splits=4, shuffle=True).split(X):
    model = Diffusion()
    
    train_dataset = CustomDataset(X[train_selector], Z[train_selector])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
    val_dataset = CustomDataset(X[val_selector], Z[val_selector])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=True, drop_last=True)
    
    trainer = Trainer(model, lr=1.0e-3, epoch=10)
    train_res = trainer.train(train_dataloader, val_dataloader, verbose=False)
    res.append(train_res)
    break

plot(res)

noise_schedule = lambda t: (1 - 2e-5) * (1 - t**2) + 1e-5    # alpha(t)
N = 10000

