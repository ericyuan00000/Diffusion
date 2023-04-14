import torch
import numpy as np
from sklearn.model_selection import KFold
from model import *
from data import *
from trainer import *
from sampler import *

X = np.zeros((10000, 2, 3))
X[:, 0, 2] = -0.5
X[:, 1, 2] = 0.5
Z = np.ones((10000, 2, 1))

model = Diffusion(n_layer=9, n_feat=256, n_atomtype=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = CustomDataset(X, Z)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
val_dataset = CustomDataset(X, Z)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=True)

trainer = Trainer(model, lr=1.0e-4, n_epoch=1000, save_model=20, save_path='model.pt', device=device)
res = trainer.train(train_dataloader, val_dataloader)

