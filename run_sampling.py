import torch
from model import *
from data import *
from trainer import *
from sampler import *

model = Diffusion(n_layer=9, n_feat=256, n_atomtype=10)
# model = Diffusion(n_layer=3, n_feat=32, n_atomtype=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('Chem C242/Diffusion/output_qm9/model.pt', map_location=device))
# model.load_state_dict(torch.load('Chem C242/Diffusion/output_h2/model.pt', map_location=device))

sampler = Sampler(model, noise_schedule=lambda t: (1 - 2e-3) * (1 - t**2) + 1e-3, n_step=10000, save_mol=1000, device=device)
X, Z = sampler.sample(n_sample=1, n_atom=10)
print(X)
print(Z.max(dim=2).indices)