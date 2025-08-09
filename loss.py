import torch
from utils import Coef
from torch.distributions import Normal
import numpy as np
from networks.egnn import remove_mean

def b_loss(x1: torch.Tensor, x2: torch.Tensor, alpha: Coef, beta: Coef, gamma: Coef, vector_field: callable, 
           center_noise: bool = False, n_particles: int = None, n_dim: int = None):
    device = x1.device
    batch_size = x1.shape[0]
    dim_size = [1 for _ in x1.shape[1:]]

    t = torch.clip(torch.rand(batch_size).to(device), 0, 1)
    
    z = torch.randn_like(x1)
    if center_noise:
        z = remove_mean(z, n_particles, n_dim)
    
    x_int = alpha.t(t).reshape(-1, *dim_size) * x1 + beta.t(t).reshape(-1, *dim_size) * x2 + gamma.t(t).reshape(-1, *dim_size) * z
    dt_x_int = alpha.dt(t).reshape(-1, *dim_size) * x1 + beta.dt(t).reshape(-1, *dim_size) * x2
    b_t = vector_field(x_int, t)
    target = dt_x_int + gamma.dt(t).reshape(-1, *dim_size) * z
    loss = 1/2 * (b_t**2).reshape(batch_size, -1).sum(-1) - (target * b_t).reshape(batch_size, -1).sum(-1)


    x_int = alpha.t(t).reshape(-1, *dim_size) * x1 + beta.t(t).reshape(-1, *dim_size) * x2 - gamma.t(t).reshape(-1, *dim_size) * z
    b_t = vector_field(x_int, t)
    target = dt_x_int - gamma.dt(t).reshape(-1, *dim_size) * z
    loss += 1/2 * (b_t**2).reshape(batch_size, -1).sum(-1) - (target * b_t).reshape(batch_size, -1).sum(-1)

    return loss.mean()


def v_loss(x1: torch.Tensor, x2: torch.Tensor, alpha: Coef, beta: Coef, gamma: Coef, vector_field: callable,
           center_noise: bool = False, n_particles: int = None, n_dim: int = None):
    device = x1.device
    batch_size = x1.shape[0]
    dim_size = [1 for _ in x1.shape[1:]]

    t = torch.clip(torch.rand(batch_size).to(device), 0, 1)

    z = torch.randn_like(x1)
    if center_noise:
        z = remove_mean(z, n_particles, n_dim)

    x_int = alpha.t(t).reshape(-1, *dim_size) * x1 + beta.t(t).reshape(-1, *dim_size) * x2 + gamma.t(t).reshape(-1, *dim_size) * z
    dt_x_int = alpha.dt(t).reshape(-1, *dim_size) * x1 + beta.dt(t).reshape(-1, *dim_size) * x2
    b_t = vector_field(x_int, t)
    target = dt_x_int
    loss = 1/2 * (b_t**2).reshape(batch_size, -1).sum(-1) - (target * b_t).reshape(batch_size, -1).sum(-1)

    return loss.mean()


def eta_loss(x1, x2, alpha: Coef, beta: Coef, gamma: Coef, score_net: callable,
             center_noise: bool = False, n_particles: int = None, n_dim: int = None):
    device = x1.device
    batch_size = x1.shape[0]
    dim_size = [1 for _ in x1.shape[1:]]

    t = torch.rand(batch_size).to(device)

    z = torch.randn_like(x1)
    if center_noise:
        z = remove_mean(z, n_particles, n_dim)

    x_int = alpha.t(t).reshape(-1, *dim_size) * x1 + beta.t(t).reshape(-1, *dim_size) * x2 + gamma.t(t).reshape(-1, *dim_size) * z
    score = score_net.score(x_int, t)
    eta = -score # note this is different from the original SI paper, which used a different eta calculation 
    # in si, eta = -score*gamma, and loss = eta**2 - e*eta
    loss = gamma.t(t).flatten() * 1/2 * (eta**2).reshape(batch_size, -1).sum(-1) - (z * eta).reshape(batch_size, -1).sum(-1)


    x_int = alpha.t(t).reshape(-1, *dim_size) * x1 + beta.t(t).reshape(-1, *dim_size) * x2 - gamma.t(t).reshape(-1, *dim_size) * z
    score = score_net.score(x_int, t)
    eta = -score
    loss += gamma.t(t).flatten() * 1/2 * (eta**2).reshape(batch_size, -1).sum(-1) + (z * eta).reshape(batch_size, -1).sum(-1)

    return loss.mean()




def tsm_loss(x1: torch.Tensor, x2: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor, alpha: Coef, beta: Coef, gamma: Coef, score_net: callable,
             center_noise: bool = False, n_particles: int = None, n_dim: int = None):
    device = x1.device
    batch_size = x1.shape[0]
    dim_size = [1 for _ in x1.shape[1:]]

    t = torch.rand(batch_size).to(device) * 0.5
    z = torch.randn_like(x1)
    if center_noise:
        z = remove_mean(z, n_particles, n_dim)

    x_int = alpha.t(t).reshape(-1, *dim_size) * x1 + beta.t(t).reshape(-1, *dim_size) * x2 + gamma.t(t).reshape(-1, *dim_size) * z
    score = score_net.score(x_int, t)
    target = s1 
    loss =  alpha.t(t).flatten() * 1/2 * (score**2).reshape(batch_size, -1).sum(-1) - (target * score).reshape(batch_size, -1).sum(-1)

    x_int = alpha.t(t).reshape(-1, *dim_size) * x1 + beta.t(t).reshape(-1, *dim_size) * x2 - gamma.t(t).reshape(-1, *dim_size) * z
    score = score_net.score(x_int, t)
    target = s1
    loss += (alpha.t(t).flatten() * 1/2 * (score**2).reshape(batch_size, -1).sum(-1) - (target * score).reshape(batch_size, -1).sum(-1))

    t = torch.rand(batch_size).to(device) * 0.5 + 0.5
    z = torch.randn_like(x1)
    if center_noise:
        z = remove_mean(z, n_particles, n_dim)
        
    x_int = alpha.t(t).reshape(-1, *dim_size) * x1 + beta.t(t).reshape(-1, *dim_size) * x2 + gamma.t(t).reshape(-1, *dim_size) * z
    score = score_net.score(x_int, t)
    target = s2
    loss += (beta.t(t).flatten() * 1/2 * (score**2).reshape(batch_size, -1).sum(-1) - (target * score).reshape(batch_size, -1).sum(-1))

    x_int = alpha.t(t).reshape(-1, *dim_size) * x1 + beta.t(t).reshape(-1, *dim_size) * x2 - gamma.t(t).reshape(-1, *dim_size) * z
    score = score_net.score(x_int, t)
    target = s2
    loss += (beta.t(t).flatten() * 1/2 * (score**2).reshape(batch_size, -1).sum(-1) - (target * score).reshape(batch_size, -1).sum(-1))

    return loss.mean()





def ti_loss(x1: torch.Tensor, x2: torch.Tensor, alpha: Coef, beta: Coef, gamma: Coef, score_net: callable,
             center_noise: bool = False, n_particles: int = None, n_dim: int = None):
    device = x1.device
    batch_size = x1.shape[0]
    dim_size = [1 for _ in x1.shape[1:]]

    t = torch.rand(batch_size).to(device)
    z = torch.randn_like(x1)
    if center_noise:
        z = remove_mean(z, n_particles, n_dim)

    x_int = alpha.t(t).reshape(-1, *dim_size) * x1 + beta.t(t).reshape(-1, *dim_size) * x2 + gamma.t(t).reshape(-1, *dim_size) * z
    t = t.requires_grad_()
    a = torch.autograd.grad(-score_net(x_int, t, False).sum(), t)[0]
    t = t.detach()

    return (a**2).mean()

