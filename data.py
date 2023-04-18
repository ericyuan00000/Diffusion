import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, atomtype=[1, 6, 7, 8, 9]):
        self.X = torch.tensor(data['R'], dtype=torch.float)
        n_sample = self.X.shape[0]
        n_atom = self.X.shape[1]
        n_atomtype = len(atomtype)
        self.Z = torch.zeros((n_sample, n_atom, n_atomtype))    # atom types, (n_sample, n_atom, n_atomtype)
        self.K1 = torch.ones((n_sample, n_atom, 1))    # node masks, (n_sample, n_atom, 1)
        self.K2 = torch.ones((n_sample, n_atom, n_atom, 1))    # edge masks, (n_sample, n_atom, n_atom, 1)
        for _sample in range(n_sample):
            for _atom in range(n_atom):
                if data['Z'][_sample, _atom, 0]>0:
                    self.Z[_sample, _atom, atomtype.index(data['Z'][_sample, _atom, 0])] = 1
                else:
                    self.K1[_sample, _atom, 0] = 0
                    self.K2[_sample, _atom, :, 0] = 0
                    self.K2[_sample, :, _atom, 0] = 0
                self.K2[_sample, _atom, _atom, 0] = 0
        

    def __len__(self):
        return self.X.shape[0]
        
        
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'Z': self.Z[idx], 'K1': self.K1[idx], 'K2': self.K2[idx]}