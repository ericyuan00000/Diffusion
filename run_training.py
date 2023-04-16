import torch
import numpy as np
from model import *
from data import *
from trainer import *
from sampler import *

torch.autograd.detect_anomaly()

train_data = np.load('/global/scratch/users/ericyuan/QM9/QM9_train.npz')
val_data = np.load('/global/scratch/users/ericyuan/QM9/QM9_val.npz')

model = Diffusion(n_layer=9, n_feat=256, n_atomtype=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = CustomDataset(train_data, n_atomtype=10)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
val_dataset = CustomDataset(val_data, n_atomtype=10)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=True)

trainer = Trainer(model, lr=1.0e-4, n_epoch=1000, save_model=20, save_path='output_qm9', device=device)
trainer.train(train_dataloader, val_dataloader)

# train_data = {'R': np.tile(np.array([[0,0,0],[0,0,1]]), [1000, 1, 1]),
#               'Z': np.tile(np.array([[1],[1]]), [1000, 1, 1])}
# val_data = {'R': np.tile(np.array([[0,0,0],[0,0,1]]), [1000, 1, 1]),
#             'Z': np.tile(np.array([[1],[1]]), [1000, 1, 1])}

# model = Diffusion(n_layer=3, n_feat=32, n_atomtype=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train_dataset = CustomDataset(train_data, n_atomtype=2)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
# val_dataset = CustomDataset(val_data, n_atomtype=2)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=True)

# trainer = Trainer(model, lr=1.0e-4, n_epoch=100, save_model=20, save_path='Chem C242/Diffusion/output_h2', device=device)
# # trainer = Trainer(model, lr=1.0e-4, n_epoch=100, save_model=20, save_path='output_h2', device=device)
# trainer.train(train_dataloader, val_dataloader)


print('done!')