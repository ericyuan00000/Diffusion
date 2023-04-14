import torch


class Sampler():
    def __init__(self, 
                 model, 
                 device,
                 noise_schedule=lambda t: (1 - 2e-5) * (1 - t**2) + 1e-5):
        self.model = model
        self.device = device
        self.noise_schedule = noise_schedule

    
    def sample(self, n_sample=100, n_step=10000):
        X_t = torch.randn((n_sample, 2, 3), device=self.device)
        Z = torch.zeros((n_sample, 2, 2), device=self.device)
        Z[:, :, 1] = 1
        H, K = self.model.encode(Z)
        for step in range(n_step):
            t_t = (1 - step / n_step) * torch.ones((X_t.shape[0], X_t.shape[1], 1), device=self.device)
            t_s = (1 - (step + 1) / n_step) * torch.ones((X_t.shape[0], X_t.shape[1], 1), device=self.device)
            alpha_t = self.noise_schedule(t_t)
            sigma_t = torch.sqrt(1 - alpha_t**2)
            alpha_s = self.noise_schedule(t_s)
            sigma_s = torch.sqrt(1 - alpha_s**2)
            alpha_ts = alpha_t / alpha_s
            sigma_ts = torch.sqrt(sigma_t**2 - alpha_ts**2 * sigma_s**2)
            
            eps_t = self.model.forward(X_t, H, K)
            mu_Q = 1 / alpha_ts * X_t - sigma_ts**2 / alpha_ts / sigma_t * eps_t
            sigma_Q = sigma_ts * sigma_s / sigma_t
            noise = torch.randn(X_t.shape)
            X_s = mu_Q + sigma_Q * noise
            X_t = X_s

            print(step, f'H2 dist: {(X_t[:, 0, :]-X_t[:, 1, :]).norm(dim=1).nanmean():.2f} ± {(X_t[:, 0, :]-X_t[:, 1, :]).norm(dim=1).nanstd():.2f}')
        return X_t