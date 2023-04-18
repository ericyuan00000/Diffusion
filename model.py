import torch
from torch import nn

class Diffusion(nn.Module):
    def __init__(self, n_layer=9, n_feat=256, atomtype=[1, 6, 7, 8, 9]):
        super(Diffusion, self).__init__()
        self.n_layer = n_layer
        self.n_feat = n_feat
        self.n_atomtype = len(atomtype)
        self.encode = nn.Linear(self.n_atomtype+1, self.n_feat)
        self.egnn_layers = nn.ModuleList([self.egnn_layer() for l in range(n_layer)])
        self.decode = nn.Linear(self.n_feat, self.n_atomtype)
        
        
    def egnn_layer(self):
        layer = nn.ModuleDict({
            'feat_message': nn.Sequential(
                nn.Linear(2*self.n_feat+2, self.n_feat),
                nn.SiLU(),
                nn.Linear(self.n_feat, self.n_feat),
                nn.SiLU()
            ),
            'feat_weight': nn.Sequential(
                nn.Linear(self.n_feat, 1),
                nn.Sigmoid()
            ),
            'feat_update': nn.Sequential(
                nn.Linear(2*self.n_feat, self.n_feat),
                nn.SiLU(),
                nn.Linear(self.n_feat, self.n_feat)
            ),
            'coord_weight': nn.Sequential(
                nn.Linear(2*self.n_feat+2, self.n_feat),
                nn.SiLU(),
                nn.Linear(self.n_feat, self.n_feat),
                nn.SiLU(),
                nn.Linear(self.n_feat, 1)
            )
        })
        return layer
    
    
    def forward(self, X, Z, K1, K2, T):
        H = self.encode(torch.cat([Z, T], dim=2)) * K1   # atomic features, (n_batch, n_atom, n_feat)

        n_batch = X.shape[0]
        n_atom = X.shape[1]
        n_feat = H.shape[2]
        
        E = torch.zeros((n_batch, n_atom, n_atom, 2*n_feat+2), device=H.device)    # edge featrues, (n_batch, n_atom, n_atom, 2*n_feat+2)
        for _layer, layer in enumerate(self.egnn_layers):
            E[:, :, :, 0:n_feat] = H[:, :, None, :].tile(1, 1, n_atom, 1)
            E[:, :, :, n_feat:2*n_feat] = H[:, None, :, :].tile(1, n_atom, 1, 1)
            D = X[:, :, None, :].tile(1, 1, n_atom, 1) - X[:, None, :, :].tile(1, n_atom, 1, 1)    # distance matrices, (n_batch, n_atom, n_atom, 3)
            E[:, :, :, [-2]] = D.norm(dim=3, keepdim=True)**2
            if _layer==0:
                E[:, :, :, [-1]] = D.norm(dim=3, keepdim=True)**2

            MH = layer['feat_message'](E.clone())    # feature messages, (n_batch, n_atom, n_atom, n_feat)
            WH = layer['feat_weight'](MH) * K2    # message weights, (n_batch, n_atom, n_atom, 1)
            H = layer['feat_update'](torch.cat([H, (MH * WH).sum(dim=2)], dim=2)) * K1 + H    # features, (n_batch, n_atom, n_feat)
            
            D = D/(D.norm(dim=3, keepdim=True)+1)
            WX = layer['coord_weight'](E.clone()) * K2    # message weights, (n_batch, n_atom, n_atom, 1)
            X = (D * WX).sum(dim=2) + X    # coordinates, (n_batch, n_atom, 3)
        
        Z = self.decode(H) * K1

        return X, Z