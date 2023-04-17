import torch
from torch import nn
from torch.optim import Adam
import random
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, 
                 model, 
                 device,
                 noise_schedule=lambda t: (1 - 2e-5) * (1 - t**2) + 1e-5,
                 lr=1.0e-3,
                 n_epoch=1000,
                 save_model=20,
                 save_path='output'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.n_epoch = n_epoch
        self.loss_func = nn.MSELoss()
        self.noise_schedule = noise_schedule    # alpha(t)
        self.save_model = save_model
        self.save_path = save_path
        self.loss_log = {'epoch':[], 'train': [], 'val':[]}

        
    def train(self, train_dataloader, val_dataloader):
        for epoch in tqdm(range(self.n_epoch)):
            self.model.train()
            train_loss = 0
            for batch_data in iter(train_dataloader):
                batch_X = batch_data['X'].to(self.device)
                batch_Z = batch_data['Z'].to(self.device)
                batch_K = batch_data['K'].to(self.device)

                n_batch = batch_X.shape[0]
                n_atom = batch_X.shape[1]
                n_atomtype = batch_Z.shape[2]

                batch_t = torch.rand(1, device=self.device).tile((n_batch, n_atom, 1))
                batch_alpha = self.noise_schedule(batch_t)  # alpha(t), weight of data
                batch_sigma = torch.sqrt(1 - batch_alpha**2)  # sigma(t), weight of noise
                batch_epsilon = torch.randn((n_batch, n_atom, 3+n_atomtype), device=self.device)  # noise
                batch_X = batch_alpha * batch_X + batch_sigma * batch_epsilon[:, :, 0:3]
                batch_Z = batch_alpha * batch_Z + batch_sigma * batch_epsilon[:, :, 3:3+n_atomtype]
                
                pred_epsilon = self.model.forward(batch_X, batch_Z, batch_K, batch_t)
                loss = self.loss_func(pred_epsilon, batch_epsilon)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.detach().cpu().item()/len(train_dataloader)
            
            self.model.eval()
            val_loss = 0
            for batch_data in iter(val_dataloader):
                batch_X = batch_data['X'].to(self.device)
                batch_Z = batch_data['Z'].to(self.device)
                batch_K = batch_data['K'].to(self.device)

                n_batch = batch_X.shape[0]
                n_atom = batch_X.shape[1]
                n_atomtype = batch_Z.shape[2]

                batch_t = torch.rand(1, device=self.device).tile((n_batch, n_atom, 1))
                batch_alpha = self.noise_schedule(batch_t)  # alpha(t), weight of data
                batch_sigma = torch.sqrt(1 - batch_alpha**2)  # sigma(t), weight of noise
                batch_epsilon = torch.randn((n_batch, n_atom, 3+n_atomtype), device=self.device)  # noise
                batch_X = batch_alpha * batch_X + batch_sigma * batch_epsilon[:, :, 0:3]
                batch_Z = batch_alpha * batch_Z + batch_sigma * batch_epsilon[:, :, 3:3+n_atomtype]
                with torch.no_grad():
                    pred_epsilon = self.model.forward(batch_X, batch_Z, batch_K, batch_t)
                    loss = self.loss_func(pred_epsilon, batch_epsilon)
                val_loss += loss.detach().cpu().item()/len(val_dataloader)

            print(f'Train loss: {train_loss:.3f} - Val loss: {val_loss:.3f}')
            self.loss_log['epoch'].append(epoch+1)
            self.loss_log['train'].append(train_loss)
            self.loss_log['val'].append(val_loss)

            if (epoch+1)%self.save_model==0:
                torch.save(self.model.state_dict(), self.save_path+'/model.pt')
            self.record_loss()


    def record_loss(self):
        df = pd.DataFrame(self.loss_log)
        df.to_csv(self.save_path+'/log.csv', index=False)

        plt.figure()
        plt.plot(self.loss_log['epoch'], self.loss_log['train'])
        plt.plot(self.loss_log['epoch'], self.loss_log['val'])
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epoch')
        plt.ylabel('MSE loss')
        plt.savefig(self.save_path+'/log.svg')
        plt.close()
