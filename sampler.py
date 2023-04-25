import torch
import numpy as np
from ase import Atoms
from ase.visualize import view


class Sampler():
    def __init__(self, 
                 model, 
                 device,
                 time_schedule=np.linspace(1, 0, 1000),
                 noise_schedule=lambda t: (1 - 2e-5) * (1 - t**2) + 1e-5,
                 save_mol=100):
        self.model = model.to(device)
        self.device = device
        self.time_schedule = time_schedule
        self.noise_schedule = noise_schedule
        self.save_mol = save_mol

    
    def sample(self, n_sample=1, n_atom=2):
        X = torch.randn((n_sample, n_atom, 3), device=self.device)
        Z = torch.randn((n_sample, n_atom, self.model.n_atomtype+1), device=self.device)    # atom types, (n_sample, n_atom, n_atomtype)
        K1 = torch.ones((n_sample, n_atom, 1), device=self.device)    # node masks, (n_sample, n_atom, 1)
        K2 = torch.ones((n_sample, n_atom, n_atom, 1), device=self.device)    # edge masks, (n_sample, n_atom, n_atom, 1)
        K2.diagonal(1, 2).zero_()
        
        self.model.eval()
        for schedule in range(len(self.time_schedule)):
            for step in range(len(self.time_schedule[schedule]) - 1):
                t_t = self.time_schedule[schedule][step] * torch.ones((n_sample, n_atom, 1), device=self.device)
                t_s = self.time_schedule[schedule][step + 1] * torch.ones((n_sample, n_atom, 1), device=self.device)
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
                noise = torch.randn((n_sample, n_atom, 3+self.model.n_atomtype+1), device=self.device)
                XZ = mu_Q + sigma_Q * noise
                X, Z = XZ[:, :, :3], XZ[:, :, 3:]
                # X = X - X.mean(dim=1)

                if step==0 or (step+1)%self.save_mol==0:
                    print('Schedule', schedule + 1, 'Step', step + 1)
                    for sample in range(n_sample):
                        positions = (X[sample] * 1.7227699310066193).tolist()
                        numbers_onehot = Z[sample][:, :-1].argmax(dim=1).tolist()
                        numbers_charge = Z[sample][:, -1].round().int().tolist()
                        # view(Atoms(positions=positions, numbers=numbers))
                        print('positions =', positions)
                        print('numbers =', numbers_onehot)
                        # print('numbers =', numbers_charge)
        return positions, numbers_charge