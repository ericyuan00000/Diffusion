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
                 noise_schedule=lambda t: (1 - 2e-5) * (1 - t**2) + 1e-5,
                 lr=1.0e-3,
                 epoch=1000):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.epoch = epoch
        self.loss_func = nn.MSELoss()
        self.noise_schedule = noise_schedule    # alpha(t)

        
    def train(self, train_dataloader, val_dataloader):
        train_losses = []
        val_losses = []
        
        for _ in tqdm(range(self.epoch)):
            self.model.train()
            train_loss = []
            for batch_data in iter(train_dataloader):
                batch_X = batch_data['X']
                batch_Z = batch_data['Z']
                batch_H, batch_K = self.model.encode(batch_Z)
                
                batch_t = torch.rand(1).tile(batch_X.shape[0], batch_X.shape[1], 1)
                batch_alpha = self.noise_schedule(batch_t)  # alpha(t), weight of data
                batch_sigma = torch.sqrt(1 - batch_alpha**2)  # sigma(t), weight of noise
                batch_epsilon = torch.randn(batch_X.shape)  # noise
                batch_X = batch_alpha * batch_X + batch_sigma * batch_epsilon
                
                pred_epsilon = self.model.forward(batch_X, batch_H, batch_K)
                loss = self.loss_func(pred_epsilon, batch_epsilon)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.detach())
            train_losses.append(np.mean(train_loss))
            
            self.model.eval()
            val_loss = []
            for batch_data in iter(val_dataloader):
                batch_X = batch_data['X']
                batch_Z = batch_data['Z']
                batch_H, batch_K = self.model.encode(batch_Z)

                batch_t = torch.rand(1).tile(batch_X.shape[0], batch_X.shape[1], 1)
                batch_alpha = self.noise_schedule(batch_t)  # alpha(t), weight of data
                batch_sigma = torch.sqrt(1 - batch_alpha**2)  # sigma(t), weight of noise
                batch_epsilon = torch.randn(batch_X.shape)  # noise
                batch_X = batch_alpha * batch_X + batch_sigma * batch_epsilon

                pred_epsilon = self.model.forward(batch_X, batch_H, batch_K)
                loss = self.loss_func(pred_epsilon, batch_epsilon)
                val_loss.append(loss.detach())
            val_losses.append(np.mean(val_loss))
            
            print(f'Train loss: {np.mean(train_loss):.3f} - Val loss: {np.mean(val_loss):.3f}')

        return {'train_losses': train_losses, 'val_losses': val_losses}
    
        
    def evaluate(self, eval_dataloader):
        eval_loss = []
        

            
        return eval_loss
    

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
    
