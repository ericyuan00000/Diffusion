import torch
import numpy as np
from model import *
from data import *
from trainer import *
from sampler import *

model = Diffusion(n_layer=9, n_feat=256, atomtype=[1, 6, 7, 8, 9])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.load_state_dict(torch.load('/global/home/users/ericyuan/Diffusion/output_qm9/train_12/model.pt', map_location=device))
model.load_state_dict(torch.load('Chem C242/Diffusion/output_qm9/train_12/model.pt', map_location=device))

time_schedule = [np.linspace(1, 0, 10001).tolist(), np.linspace(0.5, 0, 10001).tolist(), np.linspace(0.25, 0, 10001).tolist()]
noise_schedule=lambda t: (1 - 2e-3) * (1 - t**2) + 1e-3
sampler = Sampler(model, time_schedule=time_schedule, noise_schedule=noise_schedule, save_mol=1000, device=device)
positions, numbers = sampler.sample(n_sample=1, n_atom=19)
