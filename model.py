import torch
from torch import nn

class Diffusion(nn.Module):
    def __init__(self, n_layer=9, n_feat=32, n_atomtype=2):
        super(Diffusion, self).__init__()
        self.n_layer = n_layer
        self.n_feat = n_feat
        self.n_atomtype = n_atomtype
        self.embed = nn.Linear(self.n_atomtype, self.n_feat)
        self.layers = nn.ModuleList([self.egnn() for l in range(n_layer)])
        
        
    def egnn(self):
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
        

    def encode(self, Z):
        n_batch = Z.shape[0]
        n_atom = Z.shape[1]
        
        H = self.embed(Z.float())   # atomic features, (n_batch, n_atom, n_feat)
        K = torch.ones((n_batch, n_atom, n_atom), device=H.device)    # masks, (n_batch, n_atom, n_atom)
        for n_atom_ in range(n_atom):
            K[:, n_atom_, n_atom_] = 0
        return H, K
    
    
    def forward(self, X, H, K):
        n_batch = H.shape[0]
        n_atom = H.shape[1]
        n_feat = H.shape[2]
        
        E = torch.zeros((n_batch, n_atom, n_atom, 2*n_feat+2), device=H.device)    # edge featrues, (n_batch, n_atom, n_atom, 2*n_feat+2)
        for l, layer in enumerate(self.layers):
            E[:, :, :, 0:n_feat] = H[:, :, None, :].tile(1, 1, n_atom, 1)
            E[:, :, :, n_feat:2*n_feat] = H[:, None, :, :].tile(1, n_atom, 1, 1)
            D = X[:, :, None, :].tile(1, 1, n_atom, 1) - X[:, None, :, :].tile(1, n_atom, 1, 1)    # distance matrices, (n_batch, n_atom, n_atom, 3)
            E[:, :, :, [-2]] = D.norm(dim=3, keepdim=True)**2
            if l==0:
                E[:, :, :, [-1]] = D.norm(dim=3, keepdim=True)**2

            MH = layer['feat_message'](E.clone())    # feature messages, (n_batch, n_atom, n_atom, n_feat)
            WH = layer['feat_weight'](MH) * K[:, :, :, None]    # message weights, (n_batch, n_atom, n_atom, 1)
            H = layer['feat_update'](torch.cat([H, (MH*WH).sum(dim=2)], dim=2)) + H    # features, (n_batch, n_atom, n_feat)
            
            D = D/(D.norm(dim=3, keepdim=True)+1)
            WX = self.layers[l]['coord_weight'](E.clone()) * K[:, :, :, None]    # message weights, (n_batch, n_atom, n_atom, 1)
            X = (D * WX).sum(dim=2) + X    # coordinates, (n_batch, n_atom, 3)

        return torch.cat([X, H], dim=2)