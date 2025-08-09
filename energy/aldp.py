import torch
from torch import nn
import numpy as np

from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools import testsystems

import energy.openmm_interface as omi
import multiprocessing as mp


from networks.egnn import remove_mean

class AldpBoltzmann(nn.Module):
    def __init__(self, temperature=300, env='vacuum', n_threads=8, sample_path=None, device='cpu', score_path=None, lamb=1.0, scaling=1.0):
        super(AldpBoltzmann, self).__init__()

        ndim = 66
        self.device = torch.device(device)  # Set device
        
        # System setup
        if env == 'vacuum':
            self.system = testsystems.AlanineDipeptideVacuum(constraints=None)
        elif env == 'implicit':
            self.system = testsystems.AlanineDipeptideImplicit(constraints=None)
        else:
            raise NotImplementedError('This environment is not implemented.')
        
        if lamb != 1.0:
            for force in self.system.system.getForces():
                if force.__class__.__name__ == 'CustomGBForce':
                    for idx in range(force.getNumParticles()):
                        charge, sigma, epsilon = force.getParticleParameters(idx)
                        force.setParticleParameters(idx, (charge*lamb, sigma, epsilon))
        
        # Enable CUDA in OpenMM
        self.platform = mm.Platform.getPlatformByName('CUDA') if device == 'cuda' else mm.Platform.getPlatformByName('CPU')
        
        self.openmm_energy = omi.OpenMMEnergyInterfaceParallel.apply
        self.regularize_energy = omi.regularize_energy

        energy_cut = torch.tensor(1.e+8, device=self.device)
        energy_max = torch.tensor(1.e+20, device=self.device)

        # Multiprocessing is CPU-based, consider alternative approaches for GPU parallelism
        self.pool = mp.Pool(n_threads, omi.OpenMMEnergyInterfaceParallel.var_init,
                            (self.system, temperature))
        
        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos.to(self.device), self.pool)[:, 0],  # Move input tensor to GPU
            energy_cut, energy_max
        )
        
        self.scaling = scaling

        # Load sample data if numpy 
        if sample_path is not None and sample_path.endswith('.npy'):
            self.sample_path = sample_path
            self.sample_data = torch.from_numpy(np.load(sample_path)).float().to(self.device)
            self.sample_data = remove_mean(self.sample_data, 22, 3) * self.scaling

            if score_path is not None:
                self.scores = torch.from_numpy(np.load(score_path)).float().to(self.device)
                self.scores = remove_mean(self.scores, 22, 3) / self.scaling
        
        else:
            # Load sample data
            self.sample_path = sample_path
            self.sample_data = torch.load(sample_path).float().to('cpu')  # Move to GPU
            self.sample_data = remove_mean(self.sample_data, 22, 3) * self.scaling

            if score_path is not None:
                self.scores = torch.load(score_path).to('cpu')
                self.scores = remove_mean(self.scores, 22, 3)  / self.scaling
        print('sample size:', self.sample_data.shape)


    def log_prob(self, x: torch.tensor):
        return -self.norm_energy(x.to(self.device) / self.scaling)  # Ensure x is on GPU
    
    def sample(self, n_samples):
        idx = np.random.choice(self.sample_data.shape[0], size=n_samples, replace=False)
        data = self.sample_data[idx].to(self.device) .reshape(-1, 22, 3)

        return data.reshape(-1, 66)

        # data augmentation
        # generate random rotation matrix
        from scipy.stats import special_ortho_group
        x = torch.from_numpy(special_ortho_group.rvs(3)).float().to(self.device)

        return (data @ x).reshape(-1, 66)

    def get_sample_and_score(self, n_samples):
        idx = np.random.choice(self.sample_data.shape[0], size=n_samples, replace=False)
        data = self.sample_data[idx].to(self.device).reshape(-1, 22, 3)
        scores = self.scores[idx].to(self.device).reshape(-1, 22, 3)

        return data.reshape(-1, 66), torch.clip(scores.reshape(-1, 66), -1e5, 1e5)
    
        # data augmentation
        # generate random rotation matrix
        from scipy.stats import special_ortho_group
        x = torch.from_numpy(special_ortho_group.rvs(3)).float().to(self.device)

        return (data @ x).reshape(-1, 66), torch.clip((scores @ x).reshape(-1, 66), -1e5, 1e5)