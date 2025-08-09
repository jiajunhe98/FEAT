from torch import nn
import numpy as np
import torch

beta = -2
gamma = 16*0.8
grid = 16

def compute_U(phi, beta, gamma, grid):
    phi = phi.reshape(phi.shape[0], grid, grid)
    quadratic = ((beta / 2) + 4) * torch.sum(phi ** 2, (-1, -2))
    quartic = (gamma / 16) * torch.sum(phi ** 4, (-1, -2))

    # Interaction term computed once per neighbor pair
    dims = list(range(phi.ndim))
    interaction = 0.0
    for dim in dims[1:]:
        interaction += torch.sum(phi * torch.roll(phi, -1, dim), (-1, -2))
    interaction_term = -2 * interaction

    U = quadratic + quartic + interaction_term
    return U

class phi4():
    def __init__(self, positive, sample_path, device, beta=beta, gamma=gamma, grid=grid):
        self.beta = beta
        self.gamma = gamma
        self.grid = grid

        self.positive = positive
        self.device = device

        self.sample_path = sample_path
        self.sample_data = torch.load(sample_path).to(device)
    
    def log_prob(self, phi):
        phi = phi.reshape(phi.shape[0], grid, grid)
        U = compute_U(phi, self.beta, self.gamma, self.grid)

        if self.positive:
            umbrella = lambda x:  (x.reshape(x.shape[0], grid**2).mean(-1) - 0.6) ** 2 / 2 * 10
        else:
            umbrella = lambda x:  (x.reshape(x.shape[0], grid**2).mean(-1) + 0.3) ** 2 / 2 * 10

        # if self.positive:
        #     avg = torch.mean(phi, (-1, -2))
        #     mask = (avg > 0)
        #     U = torch.where(mask, U, torch.ones_like(U)*1e8, )
        return -U - umbrella(phi)

    def score(self, x: torch.Tensor):
        with torch.enable_grad():
            x.requires_grad = True
            logp = self.log_prob(x)
            score = torch.autograd.grad(logp.sum(), x)[0]
        return score 
    
    def sample(self, n_samples):
        idx = np.random.choice(self.sample_data.shape[0], size=n_samples, replace=False)
        data = self.sample_data[idx].to(self.device)
        return  data 
    
    def get_sample_and_score(self, n_samples):
        samples = self.sample(n_samples)
        with torch.enable_grad():
            samples.requires_grad = True
            logp = self.log_prob(samples)
            score = torch.autograd.grad(logp.sum(), samples)[0]
        return samples.detach(), score