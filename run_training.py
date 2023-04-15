import torch
import numpy as np
from model import *
from data import *
from trainer import *
from sampler import *

# train_data = np.load('/global/scratch/users/ericyuan/QM9/QM9_train.npz')
# val_data = np.load('/global/scratch/users/ericyuan/QM9/QM9_val.npz')
train_data = {'R': np.tile(np.array([[0,0,0],[0,0,1]]), [1000, 1, 1]),
              'Z': np.tile(np.array([[1],[1]]), [1000, 1, 1])}
val_data = {'R': np.tile(np.array([[0,0,0],[0,0,1]]), [100, 1, 1]),
            'Z': np.tile(np.array([[1],[1]]), [100, 1, 1])}

model = Diffusion(n_layer=9, n_feat=256, n_atomtype=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = CustomDataset(train_data)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
val_dataset = CustomDataset(val_data)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=True)

trainer = Trainer(model, lr=1.0e-4, n_epoch=1000, save_model=20, save_path='output', device=device)
res = trainer.train(train_dataloader, val_dataloader)

# X = np.zeros((1000, 3, 3))
# X[:, 0, 2] = -0.5
# X[:, 1, 2] = 0.5
# Z = np.ones((1000, 3, 1))
# Z[:, 2, 0] = 0

# model = Diffusion(n_layer=3, n_feat=32, n_atomtype=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train_dataset = CustomDataset(X, Z)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
# val_dataset = CustomDataset(X, Z)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=True)

# trainer = Trainer(model, lr=1.0e-4, n_epoch=100, save_model=20, save_path='Chem C242/Diffusion/output', device=device)
# res = trainer.train(train_dataloader, val_dataloader)

