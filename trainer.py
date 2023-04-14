import torch
from torch import nn
from torch.optim import Adam
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, 
                 model, 
                 device,
                 noise_schedule=lambda t: (1 - 2e-5) * (1 - t**2) + 1e-5,
                 lr=1.0e-3,
                 n_epoch=1000,
                 save_model=20,
                 save_path='model.pt'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.n_epoch = n_epoch
        self.loss_func = nn.MSELoss()
        self.noise_schedule = noise_schedule    # alpha(t)
        self.save_model = save_model
        self.save_path = save_path

        
    def train(self, train_dataloader, val_dataloader):
        train_losses = []
        val_losses = []
        
        for epoch in tqdm(range(self.n_epoch)):
            self.model.train()
            train_loss = 0
            for batch_data in iter(train_dataloader):
                batch_X = batch_data['X'].to(self.device)
                batch_Z = batch_data['Z'].to(self.device)
                batch_H, batch_K = self.model.encode(batch_Z)

                n_batch = batch_H.shape[0]
                n_atom = batch_H.shape[1]
                n_feat = batch_H.shape[2]
                
                batch_t = torch.rand(1, device=self.device).tile(n_batch, n_atom, 1)
                batch_alpha = self.noise_schedule(batch_t)  # alpha(t), weight of data
                batch_sigma = torch.sqrt(1 - batch_alpha**2)  # sigma(t), weight of noise
                batch_epsilon = torch.randn((n_batch, n_atom, 3+n_feat), device=self.device)  # noise
                batch_X = batch_alpha * batch_X + batch_sigma * batch_epsilon[:, :, 0:3]
                batch_H = batch_alpha * batch_H + batch_sigma * batch_epsilon[:, :, 3:3+n_feat]
                # batch_H = torch.cat([batch_H, batch_t], dim=2)
                
                pred_epsilon = self.model.forward(batch_X, batch_H, batch_K)
                loss = self.loss_func(pred_epsilon, batch_epsilon)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.detach()/len(train_dataloader)
            train_losses.append(train_loss)
            
            self.model.eval()
            val_loss = 0
            for batch_data in iter(val_dataloader):
                batch_X = batch_data['X'].to(self.device)
                batch_Z = batch_data['Z'].to(self.device)
                batch_H, batch_K = self.model.encode(batch_Z)

                n_batch = batch_H.shape[0]
                n_atom = batch_H.shape[1]
                n_feat = batch_H.shape[2]

                batch_t = torch.rand(1, device=self.device).tile(batch_X.shape[0], batch_X.shape[1], 1)
                batch_alpha = self.noise_schedule(batch_t)  # alpha(t), weight of data
                batch_sigma = torch.sqrt(1 - batch_alpha**2)  # sigma(t), weight of noise
                batch_epsilon = torch.randn((n_batch, n_atom, 3+n_feat), device=self.device)  # noise
                batch_X = batch_alpha * batch_X + batch_sigma * batch_epsilon[:, :, 0:3]
                batch_H = batch_alpha * batch_H + batch_sigma * batch_epsilon[:, :, 3:3+n_feat]
                # batch_H = torch.cat([batch_H, batch_t], dim=2)

                pred_epsilon = self.model.forward(batch_X, batch_H, batch_K)
                loss = self.loss_func(pred_epsilon, batch_epsilon)
                val_loss += loss.detach()/len(val_dataloader)
            val_losses.append(val_loss)

            print(f'Train loss: {train_loss:.3f} - Val loss: {val_loss:.3f}')

            if epoch%self.save_model==0:
                torch.save(self.model.state_dict(), self.save_path)

        torch.save(self.model.state_dict(), self.save_path)

        return {'train_losses': train_losses, 'val_losses': val_losses}

    

def plot(res):
    plt.figure()
    plt.plot(np.mean([res[i]['train_losses'] for i in range(len(res))], axis=0), zorder=12)
    plt.plot(np.mean([res[i]['val_losses'] for i in range(len(res))], axis=0), zorder=11)
    for i in range(len(res)):
        plt.plot(res[i]['train_losses'], color='tab:blue', alpha=0.3)
        plt.plot(res[i]['val_losses'], color='tab:orange', alpha=0.3)
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    
