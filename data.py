import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import one_hot

class CustomDataset(Dataset):
    def __init__(self, data, atomtype=[1, 6, 7, 8, 9]):
        self.X = torch.tensor(data['R'], dtype=torch.float)
        print(self.X.shape)

        n_sample = self.X.shape[0]
        n_atom = self.X.shape[1]
        self.Z = LabelEncoder().fit([0] + atomtype).transform(data['Z'].flatten()).reshape(n_sample, n_atom)    # atom types, (n_sample, n_atom)
        self.Z = one_hot(torch.tensor(self.Z, dtype=torch.long))[:, :, 1:].float()    # atom types, (n_sample, n_atom, n_atomtype)
        print(self.Z.shape)

        self.K1 = (self.Z>0).any(dim=2).unsqueeze(2)    # node masks, (n_sample, n_atom, 1)
        print(self.K1.shape)

        self.K2 = (self.K1 * self.K1.permute(0, 2, 1)).unsqueeze(3)   # edge masks, (n_sample, n_atom, n_atom, 1)
        self.K2.diagonal(dim1=1, dim2=2).zero_()
        print(self.K2.shape)
        

    def __len__(self):
        return self.X.shape[0]
        
        
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'Z': self.Z[idx], 'K1': self.K1[idx], 'K2': self.K2[idx]}