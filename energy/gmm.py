import torch
import numpy as np
from matplotlib import pyplot as plt
import itertools


class GMM(torch.nn.Module):
    def __init__(self, dim, n_mixes, loc_scaling, log_var_scaling=0.1, mean_shift=0,  seed=0,
                 n_test_set_samples=1000, device="cpu"):
        super(GMM, self).__init__()
        torch.manual_seed(seed)

        self.seed = seed
        self.n_mixes = n_mixes
        self.dim = dim
        self.n_test_set_samples = n_test_set_samples

        mean = (torch.rand((n_mixes, dim)) - 0.5)*2 * loc_scaling + mean_shift
        log_var = torch.ones((n_mixes, dim)) * log_var_scaling

        self.register_buffer("cat_probs", torch.ones(n_mixes))
        self.register_buffer("locs", mean)
        self.register_buffer("scale_trils", torch.diag_embed(torch.nn.functional.softplus(log_var)))
        self.device = device
        self.to(self.device)

        self.call_time = 0

    def to(self, device):
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
        else:
            self.cpu()

    @property
    def distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs.to(self.device))
        com = torch.distributions.MultivariateNormal(self.locs.to(self.device),
                                                     scale_tril=self.scale_trils.to(self.device),
                                                     validate_args=False)
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                     component_distribution=com,
                                                     validate_args=False)

    @property
    def test_set(self) -> torch.Tensor:
        return self.sample((self.n_test_set_samples, ))

    def log_prob(self, x: torch.Tensor, count_call=True):
        log_prob = self.distribution.log_prob(x)
        mask = torch.zeros_like(log_prob)
        mask[log_prob < -1e4] = - torch.tensor(float("inf"))
        log_prob = log_prob + mask
        if count_call:
            self.call_time += x.shape[0]
        return log_prob

    def score(self, x: torch.Tensor, count_call=True):
        with torch.enable_grad():
            x.requires_grad = True
            logp = self.log_prob(x, count_call)
            score = torch.autograd.grad(logp.sum(), x)[0]
        return score 

    def sample(self, shape=(1,), count_call=True):
        if count_call:
            self.call_time += shape[0]
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

def plot_contours(log_prob_func,
                  samples = None,
                  ax = None,
                  bounds = (-5.0, 5.0),
                  grid_width_n_points = 20,
                  n_contour_levels = None,
                  log_prob_min = -1000.0,
                  device='cpu',
                  plot_marginal_dims=[0, 1],
                  s=2,
                  alpha=0.6,
                  title=None,
                  plt_show=True,
                  xy_tick=True,):
    """Plot contours of a log_prob_func that is defined on 2D"""
    if ax is None:
        fig, ax = plt.subplots(1)
    x_points_dim1 = torch.linspace(bounds[0], bounds[1], grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)), device=device)
    log_p_x = log_prob_func(x_points).cpu().detach()
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    x_points_dim1 = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).cpu().numpy()
    x_points_dim2 = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).cpu().numpy()
    if n_contour_levels:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x)

    if samples is not None:
        samples = np.clip(samples.detach().cpu(), bounds[0], bounds[1])
        ax.scatter(samples[:, plot_marginal_dims[0]], samples[:, plot_marginal_dims[1]], s=s, alpha=alpha)
        ### x,y ticks
        if xy_tick:
            ax.set_xticks([-40,0,40])
            ax.set_yticks([-40,0,40])
        ### size of ticks
        ax.tick_params(axis='both', which='major', labelsize=15)
    if title:
        ax.set_title(title)
        ### size of title 
        ax.title.set_fontsize(40)
    if plt_show:
        plt.show()


