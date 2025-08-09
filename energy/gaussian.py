import torch
import numpy as np
from matplotlib import pyplot as plt
import itertools


class Gaussian(torch.nn.Module):
    def __init__(self, dim, device="cpu"):
        super(Gaussian, self).__init__()
        self.dim = dim
        self.device = device


    def to(self, device):
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
        else:
            self.cpu()

    @property
    def distribution(self):
        return torch.distributions.Normal(loc=torch.zeros((self.dim), device=self.device),
                                           scale=torch.ones((self.dim), device=self.device))

    def log_prob(self, x: torch.Tensor, count_call=True):
        log_prob = self.distribution.log_prob(x).sum(-1)
        return log_prob

    def score(self, x: torch.Tensor, count_call=True):
        with torch.enable_grad():
            x.requires_grad = True
            logp = self.log_prob(x, count_call)
            score = torch.autograd.grad(logp.sum(), x)[0]
        return score 

    def sample(self, shape=(1,), count_call=True):
        return self.distribution.sample(shape)
    
    def get_sample_and_logp(self, shape=(1,), count_call=True):
        samples = self.sample(shape, count_call)
        logp = self.log_prob(samples, count_call)
        return samples, logp
    
    def get_sample_and_score(self, shape=(1,), count_call=True):
        samples = self.sample(shape, count_call)
        with torch.enable_grad():
            samples.requires_grad = True
            logp = self.log_prob(samples, count_call)
            score = torch.autograd.grad(logp.sum(), samples)[0]
        return samples.detach(), score



from networks.egnn import remove_mean
class Gaussian_zero_center(torch.nn.Module):
    def __init__(self, n_dim, n_particles, device="cpu"):
        super(Gaussian_zero_center, self).__init__()
        self.n_dim = n_dim
        self.n_particles = n_particles
        self.device = device

        self.dim  = n_dim * n_particles
        self.scaling = 1


    def to(self, device):
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
        else:
            self.cpu()

    @property
    def distribution(self):
        return torch.distributions.Normal(loc=torch.zeros((self.dim), device=self.device),
                                           scale=torch.ones((self.dim), device=self.device))

    def log_prob(self, x: torch.Tensor, count_call=True):
        log_prob = self.distribution.log_prob(remove_mean(x, self.n_particles, self.n_dim)).sum(-1)
        return log_prob

    def score(self, x: torch.Tensor, count_call=True):
        with torch.enable_grad():
            x.requires_grad = True
            logp = self.log_prob(x, count_call)
            score = torch.autograd.grad(logp.sum(), x)[0]
        return score 

    def sample(self, shape=(1,), count_call=True):
        return remove_mean(self.distribution.sample(shape), self.n_particles, self.n_dim)
    
    def get_sample_and_logp(self, shape=(1,), count_call=True):
        samples = self.sample(shape, count_call)
        logp = self.log_prob(samples, count_call)
        return samples, logp
    
    def get_sample_and_score(self, shape=(1,), count_call=True):
        samples = self.sample(shape, count_call)
        with torch.enable_grad():
            samples.requires_grad = True
            logp = self.log_prob(samples, count_call)
            score = torch.autograd.grad(logp.sum(), samples)[0]
        return samples.detach(), remove_mean(score, self.n_particles, self.n_dim)
