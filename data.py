import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, n_atomtype=2):
        self.X = torch.tensor(data['R'], dtype=torch.float)
        self.Z = torch.nn.functional.one_hot(torch.tensor(data['Z'].squeeze(), dtype=torch.long), num_classes=n_atomtype)
        
        n_sample = self.X.shape[0]
        n_atom = self.X.shape[1]
        self.K = torch.ones((n_sample, n_atom, n_atom))    # masks, (n_sample, n_atom, n_atom)
        self.K.diagonal(1, 2).zero_()
        self.K[data['Z'].squeeze()==0] = 0
        self.K.permute(0, 2, 1)[data['Z'].squeeze()==0] = 0
        

    def __len__(self):
        return self.X.shape[0]
        
        
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'Z': self.Z[idx], 'K': self.K[idx]}