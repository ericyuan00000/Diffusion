import torch
from torch import nn
torch.manual_seed(0)
from model import *
from data import *
from trainer import *
from sampler import *

model = Diffusion(n_layer=9, n_feat=256, atomtype=[1, 6, 7, 8, 9])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.load_state_dict(torch.load('/global/home/users/ericyuan/Diffusion/output_qm9/train_12/model.pt', map_location=device))
model.load_state_dict(torch.load('Chem C242/Diffusion/output_qm9/train_12/model.pt', map_location=device))
noise_schedule=lambda t: (1 - 2e-5) * (1 - t**2) + 1e-5

data_X = torch.tensor([[ 1.24142616e-01,  1.56206807e+00,  5.06487195e-02],
       [-3.54038920e-03,  2.47970543e-02, -2.10528486e-02],
       [-1.46335708e+00, -4.56701847e-01,  1.92549007e-02],
       [-2.19930128e+00, -1.33172333e-01,  1.32273215e+00],
       [-3.60362647e+00, -7.23046833e-01,  1.35661536e+00],
       [-4.20186909e+00, -3.58984802e-01,  2.59200948e+00],
       [ 6.85054529e-01, -4.45289627e-01, -1.30682080e+00],
       [ 2.17174563e+00, -7.41934009e-01, -1.21309028e+00],
       [ 7.68999169e-02, -5.64317373e-01, -2.34625666e+00],
       [ 1.16787198e+00,  1.88478407e+00, -1.88530373e-02],
       [-4.27928505e-01,  2.02797359e+00, -7.71947424e-01],
       [-2.74733364e-01,  1.94436841e+00,  9.93360513e-01],
       [ 5.43678831e-01, -3.99467151e-01,  8.31934266e-01],
       [-1.99384127e+00, -2.77091864e-02, -8.37997589e-01],
       [-1.47389649e+00, -1.54282114e+00, -1.39707587e-01],
       [-1.64122660e+00, -5.21898912e-01,  2.18430644e+00],
       [-2.28519075e+00,  9.49171331e-01,  1.46838152e+00],
       [-4.18681843e+00, -3.40984925e-01,  5.02655870e-01],
       [-3.54968886e+00, -1.81885885e+00,  1.24727020e+00],
       [-5.09095995e+00, -7.24298978e-01,  2.61437735e+00],
       [ 2.70556508e+00,  5.27920783e-02, -6.81919875e-01],
       [ 2.31820557e+00, -1.66167459e+00, -6.33348649e-01],
       [ 2.59216955e+00, -8.77029612e-01, -2.21039693e+00]], device=device).unsqueeze(0)
data_Z = torch.tensor([[0,1,0,0,0],
       [0,1,0,0,0],
       [0,1,0,0,0],
       [0,1,0,0,0],
       [0,1,0,0,0],
       [0,0,0,1,0],
       [0,1,0,0,0],
       [0,1,0,0,0],
       [0,0,0,1,0],
       [1,0,0,0,0],
       [1,0,0,0,0],
       [1,0,0,0,0],
       [1,0,0,0,0],
       [1,0,0,0,0],
       [1,0,0,0,0],
       [1,0,0,0,0],
       [1,0,0,0,0],
       [1,0,0,0,0],
       [1,0,0,0,0],
       [1,0,0,0,0],
       [1,0,0,0,0],
       [1,0,0,0,0],
       [1,0,0,0,0]], device=device).unsqueeze(0)
K1 = torch.ones((1, 23, 1), device=device)
K2 = torch.ones((1, 23, 23, 1), device=device)
K2.diagonal(dim1=1, dim2=2).zero_()

t = torch.ones((1, 23, 1), device=device) * 0.1
# t = torch.rand(1, device=device).tile((1, 23, 1))
print(t[0, 0, 0])
alpha = noise_schedule(t)  # alpha(t), weight of data
sigma = torch.sqrt(1 - alpha**2)  # sigma(t), weight of noise
epsilon = torch.randn((1, 23, 3+5), device=device) * K1  # noise
X = alpha * data_X + sigma * epsilon[:, :, 0:3]
Z = alpha * data_Z + sigma * epsilon[:, :, 3:3+5]

with torch.no_grad():
    pred_epsilon = torch.cat(model.forward(X, Z, K1, K2, t), dim=2)
    alpha_ts = alpha / 0.999
    sigma_ts = torch.sqrt(sigma**2 - alpha_ts**2 * 0.002)
    print(pred_epsilon[:, 0:3, 3:])
    print(epsilon[:, 0:3, 3:])
    print(nn.MSELoss()(pred_epsilon[:, :, 3:], epsilon[:, :, 3:]))
    pred_X = 1 / alpha_ts * X - sigma_ts**2 / alpha_ts / sigma * epsilon[:, :, 0:3]
    pred_Z = 1 / alpha_ts * Z - sigma_ts**2 / alpha_ts / sigma * epsilon[:, :, 3:]
    print(nn.MSELoss()(pred_Z, data_Z))
    pred_X = 1 / alpha_ts * X - sigma_ts**2 / alpha_ts / sigma * pred_epsilon[:, :, 0:3]
    pred_Z = 1 / alpha_ts * Z - sigma_ts**2 / alpha_ts / sigma * pred_epsilon[:, :, 3:]
    print(nn.MSELoss()(pred_Z, data_Z))


