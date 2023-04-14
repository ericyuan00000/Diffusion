import torch
import numpy as np
from sklearn.model_selection import KFold
from model import *
from data import *
from trainer import *
from sampler import *

model = Diffusion(n_layer=9, n_feat=256, n_atomtype=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sampler = Sampler(model, device=device)
sampler.sample(n_sample=100, n_atom=2, n_step=1000)