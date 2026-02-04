from simtk import openmm as mm
from simtk.openmm import app
from simtk import unit
import torch
from torch import nn

R = 8.314e-3  

class CustomTestSystem:
    def __init__(self, pdb_path):
        pdb = app.PDBFile(pdb_path)
        ff = app.ForceField('amber99sbildn.xml', 'implicit/obc1.xml')
        # build the OpenMM System
        self.system   = ff.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds
        )
        # store topology for Simulation, and a NumPy array of the reference positions
        self.topology = pdb.topology
        self.reference_positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)


import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from simtk import unit

R = 8.314e-3  # kJ/mol/K

class SerialOpenMMEnergy(Function):
    @staticmethod
    def forward(ctx, x_flat, context, temperature):
        """
        x_flat: Tensor of shape (batch, n_atoms*3), device either cpu or cuda
        context:  an openmm.Context already initialized for your system
        temperature: float (kelvin)
        """
        # detach to numpy on CPU
        x_np = x_flat.detach().cpu().numpy()
        batch, D = x_np.shape

        # infer n_atoms
        system = context.getSystem()
        n_atoms = system.getNumParticles()
        assert D == n_atoms*3, f"Expected {n_atoms*3}, got {D}"

        kBT = R * temperature

        energies = []
        forces = []
        for coords_flat in x_np:
            xyz = coords_flat.reshape(n_atoms, 3)
            context.setPositions(xyz)
            state = context.getState(getEnergy=True, getForces=True)

            # potential energy in kJ/mol
            E = state.getPotentialEnergy().value_in_unit(unit.kilojoule/unit.mole)
            energies.append(E / kBT)

            # forces as (n_atoms,3), in kJ/mol/nm, convert and negate for grad
            f = state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule/unit.mole/unit.nanometer
                ) / kBT
            forces.append(-f)

        # to tensors on x_flat's device
        E_tensor = torch.tensor(energies, device=x_flat.device).view(batch,1)
        F_tensor = torch.tensor(np.stack(forces), device=x_flat.device)  # (batch,n_atoms,3)

        # save for backward
        ctx.save_for_backward(F_tensor)
        return E_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: (batch,1)
        (F_tensor,) = ctx.saved_tensors  # (batch,n_atoms,3)
        batch, n_atoms, _ = F_tensor.shape

        # flatten forces to (batch, n_atoms*3)
        F_flat = F_tensor.view(batch, n_atoms*3)

        # dE/dx = forces; so gradient = forces * dL/dE = forces * grad_output
        return F_flat * grad_output, None, None
    

from networks.egnn import remove_mean
class A4(nn.Module):
    def __init__(self,  temperature=300, device='cpu', scaling=1.0, lamb=1.0, sample_path=None,  score_path=None, ):
        super().__init__()
        self.temperature = temperature
        self.device = torch.device(device)
        self.scaling = scaling

        # 1) load your custom system
        tsys = CustomTestSystem('energy/AAAA.pdb')

        self.system   = tsys.system
        self.topology = tsys.topology
        for force in self.system.getForces():
            if force.__class__.__name__ == 'CustomGBForce':
                for i in range(force.getNumPerParticleParameters()):
                    print (f"Per-particle parameter {i}: {force.getPerParticleParameterName(i)}")
                for idx in range(force.getNumParticles()):
                    params = force.getParticleParameters(idx)
                    # charge, sigma, epsilon = force.getParticleParameters(idx)
                    params = list(params)
                    params[0] = params[0] * lamb  # scale charge
                    force.setParticleParameters(idx, tuple(params))

        # how many atoms?
        self.n_atoms = self.system.getNumParticles()
        self.expected_dim = self.n_atoms * 3
        print(f"System has {self.n_atoms} atoms → input vectors must be length {self.expected_dim}")

        # 2) build one OpenMM Simulation/Context on the desired platform
        integrator = mm.LangevinIntegrator(
            temperature * unit.kelvin,
            1.0 / unit.picosecond,
            1.0 * unit.femtosecond
        )
        platform = mm.Platform.getPlatformByName('CUDA') if device=='cuda' else mm.Platform.getPlatformByName('CPU')
        sim = app.Simulation(self.topology, self.system, integrator, platform)
        self.context = sim.context

        # 3) thresholds for regularization (same as before)
        self.energy_cut = torch.tensor(1e8,  device=self.device)
        self.energy_max = torch.tensor(1e20, device=self.device)

        # Load sample data
        self.sample_path = sample_path
        self.sample_data = torch.load(sample_path).float().to('cpu')  # Move to GPU
        self.sample_data = remove_mean(self.sample_data, 43, 3) * self.scaling

        if score_path is not None:
            self.scores = torch.load(score_path).to('cpu')
            self.scores = remove_mean(self.scores, 43, 3)  / self.scaling
        print('sample size:', self.sample_data.shape)

        self.sample_data = self.sample_data
        self.scores = self.scores

    def norm_energy(self, x: torch.Tensor):
        E = SerialOpenMMEnergy.apply(x.to(self.device),
                                     self.context,
                                     self.temperature)  # → (batch,1)
        # regularize as before
        E = torch.where(E < self.energy_max, E, self.energy_max)
        E = torch.where(
            E < self.energy_cut,
            E,
            torch.log(E - self.energy_cut + 1) + self.energy_cut
        )
        return E

    def log_prob(self, x):
        # scale coords if desired, then negate energy
        return -self.norm_energy(x / self.scaling).flatten()

    def sample(self, n_samples):
        idx = np.random.choice(self.sample_data.shape[0], size=n_samples, replace=False)
        data = self.sample_data[idx].to(self.device).reshape(-1, 43, 3)

        return data.reshape(-1, 43*3)


    def get_sample_and_score(self, n_samples):
        idx = np.random.choice(self.sample_data.shape[0], size=n_samples, replace=False)
        data = self.sample_data[idx].to(self.device).reshape(-1, 43, 3)
        scores = self.scores[idx].to(self.device).reshape(-1, 43, 3)

        return data.reshape(-1, 43*3), torch.clip(scores.reshape(-1, 43*3), -1e5, 1e5)
    
    def get_sample_and_score_idx(self, idx):
        data = self.sample_data[idx].to(self.device).reshape(-1, 43, 3)
        scores = self.scores[idx].to(self.device).reshape(-1, 43, 3)

        return data.reshape(-1, 43*3), torch.clip(scores.reshape(-1, 43*3), -1e5, 1e5)
