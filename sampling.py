import torch
from utils import Coef
import numpy as np
from torch.distributions import Normal
from tqdm import tqdm
from networks.egnn import remove_mean



# Controlled Jarzynski with Eular-Maruyama integration
def Jarzynski_integrate(x_init: torch.Tensor, 
                        init_logprob: torch.Tensor, 
                        eps: Coef, 
                        alpha: Coef,
                        beta: Coef,
                        gamma: Coef, 
                        vector_field: callable, 
                        score_net: callable, 
                        collect_interval: int = 10, 
                        learn_b: bool = True, 
                        times: torch.Tensor = None,
                        forward: bool = True,
                        return_A: bool = False,
                        target_logp: callable = None,
                        center_noise: bool = False,
                        n_particles: int = None,
                        n_dim: int = None,
                        ):

    t_start = times[0]
    t_end = times[-1]
    
    if forward:
        assert t_start < t_end
    else:
        assert t_start > t_end

    device = x_init.device
    batch_size = x_init.shape[0]
    dim_size = [1 for _ in x_init.shape[1:]]

    # initialize
    A = torch.zeros(batch_size).to(device)
    t = torch.zeros(batch_size).to(device) + t_start
    
    # sample
    x = x_init
    Xs = [x]
    weighted_Xs = []
    ESS = []

    A -= init_logprob
    for i in range(len(times)-1):

        step_size = np.abs((times[i+1] - times[i]).item())

        if i == 0:
            print (x.shape)
            vf = vector_field(x, t) 
            s = score_net.score(x, t)
            if not learn_b:
                vf = vf - gamma.tdt(t).reshape(-1, *dim_size) * s
        std = torch.sqrt(2 * step_size * eps.t(t).reshape(-1, *dim_size))
        if forward:
            noise = torch.randn_like(x)
            if center_noise:
                noise = remove_mean(noise, n_particles, n_dim)
            x_new = x + step_size * vf +  step_size * eps.t(t).reshape(-1, *dim_size) * s + std * noise
            t_new = t*0 + times[i+1]
            A -= Normal(x+step_size*vf+step_size*eps.t(t).reshape(-1, *dim_size)*s, std, validate_args=False).log_prob(x_new).sum(-1)
            vf_new = vector_field(x_new, t_new)
            if not learn_b:
                vf_new = vf_new - gamma.tdt(t_new).reshape(-1, *dim_size) * s
            s_new = score_net.score(x_new, t_new)
            std_new = torch.sqrt(2 * step_size * eps.t(t_new).reshape(-1, *dim_size))
            A += Normal(x_new-step_size*vf_new+step_size*eps.t(t_new).reshape(-1, *dim_size) * s_new, std_new, validate_args=False).log_prob(x).sum(-1)
        else:
            noise = torch.randn_like(x)
            if center_noise:
                noise = remove_mean(noise, n_particles, n_dim)
            x_new = x - step_size * vf +  step_size * eps.t(t).reshape(-1, *dim_size) * s + std * noise
            t_new = t*0 + times[i+1]
            A -= Normal(x-step_size*vf+step_size*eps.t(t).reshape(-1, *dim_size)*s, std, validate_args=False).log_prob(x_new).sum(-1)
            vf_new = vector_field(x_new, t_new)
            if not learn_b:
                vf_new = vf_new - gamma.tdt(t_new).reshape(-1, *dim_size) * s
            s_new = score_net.score(x_new, t_new)
            std_new = torch.sqrt(2 * step_size * eps.t(t_new).reshape(-1, *dim_size))
            A += Normal(x_new+step_size*vf_new+step_size*eps.t(t_new).reshape(-1, *dim_size) * s_new, std_new, validate_args=False).log_prob(x).sum(-1)

        x = x_new
        t = t_new
        vf = vf_new
        s = s_new

        if (i+1) % collect_interval == 0 or i == len(times)-2:
            weighted_Xs.append((x, t.clone()))
        Xs.append(x)

    if return_A:
        A_final = A + target_logp(x).flatten()
        return x, Xs, A_final
    
    return x, Xs, 



# # Controlled Jarzynski 
def Jarzynski_integrate_ODE(x1: torch.Tensor, vector_field: callable, n_steps: int = 100, forward: bool = True, use_pyg: bool = False, calculate_div: bool = True):

    if forward:
        t_start = 0
        t_end = 1
    else:
        t_start = 1
        t_end = 0
    
    step_size = (t_end - t_start) / n_steps
    if use_pyg:
        device = x1.x.device
        batch_size = x1.batch_size
        dim_size = [1 for _ in x1.x.shape[1:]]
    else:
        device = x1.device 
        batch_size = x1.shape[0]  
        dim_size = [1 for _ in x1.shape[1:]]

    # initialize
    A = torch.zeros(batch_size).to(device)
    t = torch.zeros(batch_size).to(device) + t_start
    
    # sample
    x = x1
    Xs = [x]
    if use_pyg:
        x.input = x.x
    for i in range(n_steps-1):
        vf = vector_field(x, t).detach()
        if calculate_div:
            A = A + vector_field.div(x, t).detach() * step_size 
        if use_pyg:
            x.input = x.input.detach() + step_size * vf.detach()
        else:
            x = x.detach() + step_size * vf.detach()
        
        t += step_size
        Xs.append(x.clone())

    
    return x, Xs, A

