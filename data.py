import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, atomtype=[1, 6, 7, 8, 9]):
        self.X = torch.tensor(data['R'], dtype=torch.float)
        print(self.X.shape)
        n_sample = self.X.shape[0]
        n_atom = self.X.shape[1]
        n_atomtype = len(atomtype)
        self.Z = torch.zeros((n_sample, n_atom, n_atomtype))    # atom types, (n_sample, n_atom, n_atomtype)
        print(self.Z.shape)
        self.K1 = torch.ones((n_sample, n_atom, 1))    # node masks, (n_sample, n_atom, 1)
        self.K1[data['Z']==0] = 0
        print(self.K1.shape)
        self.K2 = (self.K1 * self.K1.permute(0, 2, 1)).unsqueeze(3)   # edge masks, (n_sample, n_atom, n_atom, 1)
        self.K2.diagonal(dim1=1, dim2=2).zero_()
        print(self.K2.shape)
        

    def __len__(self):
        return self.X.shape[0]
        
        
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'Z': self.Z[idx], 'K1': self.K1[idx], 'K2': self.K2[idx]}