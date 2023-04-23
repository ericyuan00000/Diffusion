import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import one_hot

class CustomDataset(Dataset):
    def __init__(self, data, n_atomtype=10, Zonehot_scale=1, Zcharge_scale=1):
        self.X = torch.tensor(data['R'], dtype=torch.float)
        self.X_scale = data['R'][data['R'].nonzero()].std()
        self.X = self.X / self.X_scale
        print(self.X.shape)

        self.Zonehot = one_hot(torch.tensor(data['Z'].squeeze(), dtype=torch.long), num_classes=n_atomtype).float()    # atom types, (n_sample, n_atom, n_atomtype)
        print(self.Zonehot.shape)
        self.Zcharge = torch.tensor(data['Z'], dtype=torch.float)    # atom types, (n_sample, n_atom, 1)
        print(self.Zcharge.shape)
        self.Zonehot_scale = Zonehot_scale
        self.Zcharge_scale = Zcharge_scale
        self.Z = torch.cat([self.Zonehot * self.Zonehot_scale, self.Zcharge * self.Zcharge_scale], dim=2)    # atom types, (n_sample, n_atom, n_atomtype+1)
        print(self.Z.shape)

        self.K1 = torch.tensor(data['Z'] > 0)    # node masks, (n_sample, n_atom, 1)
        print(self.K1.shape)

        self.K2 = (self.K1 * self.K1.permute(0, 2, 1)).unsqueeze(3)   # edge masks, (n_sample, n_atom, n_atom, 1)
        self.K2.diagonal(dim1=1, dim2=2).zero_()
        print(self.K2.shape)
        

    def __len__(self):
        return self.X.shape[0]
        
        
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'Z': self.Z[idx], 'K1': self.K1[idx], 'K2': self.K2[idx]}