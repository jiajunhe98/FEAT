import openmm as mm
from bgmol.datasets import ChignolinOBC2PT
import energy.openmm_interface as omi
import multiprocessing as mp

import torch
from torch import nn
import numpy as np

from networks.egnn import remove_mean





class Chignolin(nn.Module):
    def __init__(self, temperature=300, device='cpu', lamb=1.0, scaling=1.0, sample_path=None, score_path=None, n_threads=1):
        super().__init__()

        self.device = torch.device(device)  # Set device
        
        # System setup
        dataset = ChignolinOBC2PT(read=True, download=False)
        self.system = dataset._system
        
        for force in self.system.system.getForces():
            if force.__class__.__name__ == 'CustomGBForce':
                for idx in range(force.getNumParticles()):
                    charge, sigma, epsilon = force.getParticleParameters(idx)
                    force.setParticleParameters(idx, (charge*lamb, sigma, epsilon))
        
        # Enable CUDA in OpenMM
        self.platform = mm.Platform.getPlatformByName('CUDA') if device == 'cuda' else mm.Platform.getPlatformByName('CPU')
        
        self.openmm_energy = omi.OpenMMEnergyInterfaceParallel.apply
        self.regularize_energy = omi.regularize_energy

        energy_cut = torch.tensor(1.e+10, device=self.device)
        energy_max = torch.tensor(1.e+20, device=self.device)

        # Multiprocessing is CPU-based, consider alternative approaches for GPU parallelism
        self.pool = mp.Pool(n_threads, omi.OpenMMEnergyInterfaceParallel.var_init,
                            (self.system, temperature))
        
        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos.to(self.device), self.pool)[:, 0],  # Move input tensor to GPU
            energy_cut, energy_max
        )
        self.scaling = scaling


        # Load sample data
        if sample_path is not None:
            self.sample_data = torch.load(sample_path).float().to('cpu')  # Move to GPU
            # indices = torch.randperm(self.sample_data.shape[0])[:5]
            # self.sample_data = remove_mean(self.sample_data, 175, 3)[:1] * self.scaling
            self.sample_data = remove_mean(self.sample_data, 175, 3) * self.scaling

        if score_path is not None:
            self.scores = torch.load(score_path).to('cpu')
            
            # self.scores = remove_mean(self.scores, 175, 3)[:1]  / self.scaling
            self.scores = remove_mean(self.scores, 175, 3)  / self.scaling
        
            print(self.sample_data.shape, self.scores.shape)

    def log_prob(self, x: torch.tensor):
        return -self.norm_energy(x.to(self.device) / self.scaling)  # Ensure x is on GPU
    
    
    def sample(self, n_samples):
        idx = np.random.choice(self.sample_data.shape[0], size=n_samples, replace=False)
        data = self.sample_data[idx].to(self.device) .reshape(-1, 175, 3)

        return data.reshape(-1, 175*3)

    def get_sample_and_score(self, n_samples):
        idx = np.random.choice(self.sample_data.shape[0], size=n_samples, replace=False)
        data = self.sample_data[idx].to(self.device).reshape(-1, 175, 3)
        scores = self.scores[idx].to(self.device).reshape(-1, 175, 3)

        return data.reshape(-1, 175*3), torch.clip(scores.reshape(-1, 175*3), -1e5, 1e5)
