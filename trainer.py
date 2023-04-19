import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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
        self.scheduler = ReduceLROnPlateau(self.optimizer, verbose=True)

        self.n_epoch = n_epoch
        self.loss_func = nn.MSELoss()
        self.noise_schedule = noise_schedule    # alpha(t)
        self.save_model = save_model
        i = 1
        while True:
            if f'train_{i:02}' in os.listdir(save_path):
                i += 1
            else:
                self.save_path = f'{save_path}/train_{i:02}'
                os.mkdir(self.save_path)
                break
        self.loss_log = {'epoch':[], 'train': [], 'val':[]}

        
    def train(self, train_dataloader, val_dataloader):
        for epoch in tqdm(range(self.n_epoch)):
            self.model.train()
            train_loss = []
            for batch_data in iter(train_dataloader):
                batch_X = batch_data['X'].to(self.device)    # coordinates, (n_batch, n_atom, 3)
                batch_Z = batch_data['Z'].to(self.device)    # atom types, (n_batch, n_atom, n_atomtype)
                batch_K1 = batch_data['K1'].to(self.device)    # node masks, (n_batch, n_atom, 1)
                batch_K2 = batch_data['K2'].to(self.device)    # node masks, (n_batch, n_atom, n_atom)

                n_batch = batch_X.shape[0]
                n_atom = batch_X.shape[1]
                n_atomtype = batch_Z.shape[2]

                batch_t = torch.rand(1, device=self.device).tile((n_batch, n_atom, 1))
                batch_alpha = self.noise_schedule(batch_t)  # alpha(t), weight of data
                batch_sigma = torch.sqrt(1 - batch_alpha**2)  # sigma(t), weight of noise
                batch_epsilon = torch.randn((n_batch, n_atom, 3+n_atomtype), device=self.device) * batch_K1  # noise
                batch_X = batch_alpha * batch_X + batch_sigma * batch_epsilon[:, :, 0:3]
                batch_Z = batch_alpha * batch_Z + batch_sigma * batch_epsilon[:, :, 3:3+n_atomtype]

                try:
                    pred_epsilon = torch.cat(self.model.forward(batch_X, batch_Z, batch_K1, batch_K2, batch_t), dim=2)
                    loss = self.loss_func(pred_epsilon, batch_epsilon)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                    self.optimizer.step()
                    train_loss.append(loss.item())
                except:
                    print('RuntimeError: The total norm for gradients is non-finite, so it cannot be clipped.')
                    print(pred_epsilon)
                    print(batch_epsilon)
            train_loss = np.mean(train_loss)
            
            self.model.eval()
            val_loss = []
            for batch_data in iter(val_dataloader):
                batch_X = batch_data['X'].to(self.device)    # coordinates, (n_batch, n_atom, 3)
                batch_Z = batch_data['Z'].to(self.device)    # atom types, (n_batch, n_atom, n_atomtype)
                batch_K1 = batch_data['K1'].to(self.device)    # node masks, (n_batch, n_atom, 1)
                batch_K2 = batch_data['K2'].to(self.device)    # node masks, (n_batch, n_atom, n_atom)

                n_batch = batch_X.shape[0]
                n_atom = batch_X.shape[1]
                n_atomtype = batch_Z.shape[2]

                batch_t = torch.rand(1, device=self.device).tile((n_batch, n_atom, 1))
                batch_alpha = self.noise_schedule(batch_t)  # alpha(t), weight of data
                batch_sigma = torch.sqrt(1 - batch_alpha**2)  # sigma(t), weight of noise
                batch_epsilon = torch.randn((n_batch, n_atom, 3+n_atomtype), device=self.device) * batch_K1  # noise
                batch_X = batch_alpha * batch_X + batch_sigma * batch_epsilon[:, :, 0:3]
                batch_Z = batch_alpha * batch_Z + batch_sigma * batch_epsilon[:, :, 3:3+n_atomtype]

                with torch.no_grad():
                    pred_epsilon = torch.cat(self.model.forward(batch_X, batch_Z, batch_K1, batch_K2, batch_t), dim=2)
                    loss = self.loss_func(pred_epsilon, batch_epsilon)
                val_loss.append(loss.item())
            val_loss = np.mean(val_loss)

            print(f'Train loss: {train_loss:.3f} - Val loss: {val_loss:.3f}')
            self.loss_log['epoch'].append(epoch+1)
            self.loss_log['train'].append(train_loss)
            self.loss_log['val'].append(val_loss)

            if (epoch+1)%self.save_model==0:
                torch.save(self.model.state_dict(), f'{self.save_path}/model.pt')
            self.record_loss()

            self.scheduler.step(val_loss)


    def record_loss(self):
        df = pd.DataFrame(self.loss_log)
        df.to_csv(f'{self.save_path}/log.csv', index=False)

        plt.figure()
        plt.plot(self.loss_log['epoch'], self.loss_log['train'])
        plt.plot(self.loss_log['epoch'], self.loss_log['val'])
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epoch')
        plt.ylabel('MSE loss')
        plt.yscale('log')
        plt.savefig(f'{self.save_path}/log.svg')
        plt.close()
