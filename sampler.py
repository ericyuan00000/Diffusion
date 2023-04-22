import torch
import numpy as np
from ase import Atoms
from ase.visualize import view


class Sampler():
    def __init__(self, 
                 model, 
                 device,
                 noise_schedule=lambda t: (1 - 2e-5) * (1 - t**2) + 1e-5,
                 n_step=1000,
                 save_mol=100):
        self.model = model.to(device)
        self.device = device
        self.noise_schedule = noise_schedule
        self.n_step = n_step
        self.save_mol = save_mol

    
    def sample(self, n_sample=1, n_atom=2):
        X = torch.randn((n_sample, n_atom, 3), device=self.device)
        Z = torch.randn((n_sample, n_atom, self.model.n_atomtype), device=self.device)    # atom types, (n_sample, n_atom, n_atomtype)
        K1 = torch.ones((n_sample, n_atom, 1), device=self.device)    # node masks, (n_sample, n_atom, 1)
        K2 = torch.ones((n_sample, n_atom, n_atom, 1), device=self.device)    # edge masks, (n_sample, n_atom, n_atom, 1)
        K2.diagonal(1, 2).zero_()
        
        self.model.eval()
        for _step in range(self.n_step):
            t_t = (1 - _step / self.n_step) * torch.ones((n_sample, n_atom, 1), device=self.device)
            t_s = (1 - (_step + 1) / self.n_step) * torch.ones((n_sample, n_atom, 1), device=self.device)
            alpha_t = self.noise_schedule(t_t)
            sigma_t = torch.sqrt(1 - alpha_t**2)
            alpha_s = self.noise_schedule(t_s)
            sigma_s = torch.sqrt(1 - alpha_s**2)
            alpha_ts = alpha_t / alpha_s
            sigma_ts = torch.sqrt(sigma_t**2 - alpha_ts**2 * sigma_s**2)
            with torch.no_grad():
                epsilon_t = torch.cat(self.model.forward(X, Z, K1, K2, t_t), dim=2)
            mu_Q = 1 / alpha_ts * torch.cat([X, Z], dim=2) - sigma_ts**2 / alpha_ts / sigma_t * epsilon_t
            sigma_Q = sigma_ts * sigma_s / sigma_t
            noise = torch.randn((n_sample, n_atom, 3+self.model.n_atomtype), device=self.device)
            XZ = mu_Q + sigma_Q * noise
            X, Z = XZ[:, :, 0:3], XZ[:, :, 3:3+self.model.n_feat]

            if _step==0 or (_step+1)%self.save_mol==0:
                for _sample in range(n_sample):
                    positions = list(X[_sample].tolist())
                    numbers = []
                    for z in Z[_sample]:
                        numbers.append(self.model.atomtype[z.argmax()])
                    # view(Atoms(positions=positions, numbers=numbers))
                    print('Step', _step)
                    print('positions =', positions)
                    print('numbers =', numbers)
                    print(Z[_sample])
        return positions, numbers