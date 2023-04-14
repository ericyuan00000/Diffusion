import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, Z, n_atomtype=2):
        self.X = torch.tensor(X, dtype=torch.float)
        self.Z = torch.nn.functional.one_hot(torch.tensor(Z.squeeze(), dtype=torch.long), num_classes=n_atomtype)
        

    def __len__(self):
        return self.X.shape[0]
        
        
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'Z': self.Z[idx]}