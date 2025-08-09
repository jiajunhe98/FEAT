import numpy as np
import torch
from bgflow import Energy
from bgflow.utils import distance_vectors, distances_from_vectors
from scipy.interpolate import CubicSpline
from functools import partial
from networks.egnn import remove_mean

def sample_from_array(array, size):
    idx = np.random.choice(array.shape[0], size=size)
    return array[idx]


def lennard_jones_energy_torch(r, eps=1.0, rm=1.0, a = 0, *args, **kwargs):
    if eps == 0:
        return torch.zeros_like(r)
    b = rm * (2**(1/6) - (2-a)**(1/6))
    lj = 4 * eps * ((a + ((r-b) / rm)**6)**(-2) - (a + ((r-b) / rm)**6)**(-1))
    return lj

# def lennard_jones_energy_torch(r, eps=1.0, rm=1.0, a = 0, *args, **kwargs):
#     lj = 4 * eps * ((rm**2 / (r**2 + a * rm ** 2)) ** 6 - ((rm**2 / (r**2 + a * rm ** 2)) ** 3))
#     return lj

def cubic_spline(x_new, x, c):
    x, c = x.to(x_new.device), c.to(x_new.device)
    intervals = torch.bucketize(x_new, x) - 1
    intervals = torch.clamp(intervals, 0, len(x) - 2) # Ensure valid intervals
    # Calculate the difference from the left breakpoint of the interval
    dx = x_new - x[intervals]
    # Evaluate the cubic spline at x new
    y_new = (c[0, intervals] * dx ** 3 + \
             c[1, intervals]* dx**2 + \
             c[2, intervals]* dx +\
             c[3, intervals])
    return y_new

class LennardJonesPotential(Energy):
    def __init__(
        self,
        dim,
        n_particles,
        sample_path,
        score_path=None,
        eps=1.0,
        rm=1.0,
        oscillator=True,
        oscillator_scale=1.0,
        two_event_dims=True,
        energy_factor=1.0,
        range_min=0.3,
        range_max=2.,
        interpolation=1000,
        device="cpu",
        a = 0, # smooth parameter
        rescaling = 1.0
    ):
        """Energy for a Lennard-Jones cluster.

        Parameters
        ----------
        dim : int
            Number of degrees of freedom ( = space dimension x n_particles)
        n_particles : int
            Number of Lennard-Jones particles
        eps : float
            LJ well depth epsilon
        rm : float
            LJ well radius R_min
        oscillator : bool
            Whether to use a harmonic oscillator as an external force
        oscillator_scale : float
            Force constant of the harmonic oscillator energy
        two_event_dims : bool
            If True, the energy expects inputs with two event dimensions (particle_id, coordinate).
            Else, use only one event dimension.
        """
        if two_event_dims:
            super().__init__([n_particles, dim // n_particles])
        else:
            super().__init__(dim)
        self._n_particles = n_particles
        self.n_spatial_dim = dim // n_particles

        self._eps = eps
        self._rm = rm
        self.oscillator = oscillator
        self._oscillator_scale = oscillator_scale

        self._energy_factor = energy_factor
        
        self.range_min = range_min
        self.range_max = range_max
    
        #fit spline cubic on these ranges
        interpolate_points = torch.linspace(range_min, range_max, interpolation)
        
        es = lennard_jones_energy_torch(interpolate_points, 
                                            self._eps, self._rm, a=a
                                            )
        coeffs = CubicSpline(np.array(interpolate_points),
                                np.array(es)).c
        self.splines = partial(cubic_spline, 
                                x=interpolate_points,
                                c=torch.tensor(coeffs).float())
        self.device = device

        self.scaling = rescaling

        self.sample_path = sample_path
        self.sample_data = torch.load(sample_path)
        self.sample_data = remove_mean(self.sample_data, self._n_particles, self.n_spatial_dim).reshape(-1, self._n_particles * self.n_spatial_dim) * self.scaling

        if score_path is not None:
            self.scores = torch.load(score_path)
            self.scores = remove_mean(self.scores, self._n_particles, self.n_spatial_dim).reshape(-1, self._n_particles * self.n_spatial_dim) / self.scaling

            print(self.scores.isnan().any())
        else:
            self.scores = None


        self.to(self.device)
    
    def to(self, device):
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
        else:
            self.cpu()

    def log_prob(self, x, smooth_=False):
        if len(x.shape) == len(self.event_shape):
            x = x.unsqueeze(0)
        if x.shape[0] == 0:
            return torch.zeros([0, 1]).to(x.device)
        batch_shape = x.shape[: -len(self.event_shape)]
        x = x.view(*batch_shape, self._n_particles, self.n_spatial_dim) / self.scaling

        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self._n_particles, self.n_spatial_dim))
        )

        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm)
        
        if smooth_:
            lj_energies[dists < self.range_min] = self.splines(dists[dists < self.range_min]).squeeze(-1)
        #lj_energies = lj_energies.view(*batch_shape, -1).sum(dim=-1) * self._energy_factor
        lj_energies = lj_energies.view(*batch_shape, self._n_particles, -1).sum(dim=-1) * self._energy_factor

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=( -1)).view(*batch_shape, self._n_particles)
            lj_energies = lj_energies + osc_energies * self._oscillator_scale
        return -lj_energies.sum(-1)#[:, None]
    
    def score(self, x, smooth_=False):
        with torch.enable_grad():
            x.requires_grad = True
            logp = self.log_prob(x, smooth_)
            score = torch.autograd.grad(logp.sum(), x)[0]
        return score#[:, None]

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self.n_spatial_dim)
        return x - torch.mean(x, dim=1, keepdim=True)

    def _energy_numpy(self, x):
        x = torch.Tensor(x)
        return self._energy(x).cpu().numpy()

    def energy(self, x, smooth_=False):
        if len(x.shape) == len(self.event_shape):
            x = x.unsqueeze(0)
        if x.shape[0] == 0:
            return torch.zeros([0, 1]).to(x.device)
        batch_shape = x.shape[: -len(self.event_shape)]
        x = x.view(*batch_shape, self._n_particles, self.n_spatial_dim) / self.scaling

        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self._n_particles, self.n_spatial_dim))
        )

        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm)
        
        if smooth_:
            lj_energies[dists < self.range_min] = self.splines(dists[dists < self.range_min]).squeeze(-1)
        lj_energies = lj_energies.view(*batch_shape, self._n_particles, -1).sum(dim=-1) * self._energy_factor

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=( -1)).view(*batch_shape, self._n_particles)
            lj_energies = lj_energies + osc_energies * self._oscillator_scale
        return lj_energies.sum(-1)
    
    def sample(self, n_samples):
        idx = np.random.choice(self.sample_data.shape[0], size=n_samples, replace=False)
        
        # from scipy.stats import special_ortho_group
        # x = torch.from_numpy(special_ortho_group.rvs(3)).float().to(self.device)

        data = self.sample_data[idx].to(self.device)#.reshape(-1, self._n_particles, self.n_spatial_dim)
        return  data # (data @ x).reshape(-1, self._n_particles * self.n_spatial_dim)

    def get_sample_and_score(self, n_samples):
        if self.scores is not None:
            idx = np.random.choice(self.sample_data.shape[0], size=n_samples, replace=False)
            data = self.sample_data[idx].to(self.device).reshape(-1, self._n_particles, self.n_spatial_dim)
            scores = self.scores[idx].to(self.device).reshape(-1, self._n_particles, self.n_spatial_dim)

            return data.reshape(-1, self._n_particles* self.n_spatial_dim), torch.clip(scores.reshape(-1, self._n_particles* self.n_spatial_dim), -1e5, 1e5)
        samples = self.sample(n_samples)
        score = self.score(samples)
        return samples.detach(), score.detach()


    