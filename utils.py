import torch 
import numpy as np
import matplotlib.pyplot as plt

class Coef:
    def __init__(self, coef, a=1.0, b=0.001):
        self.coef = coef
        self.a = a
        self.b = b
    
    def t(self, t):
        if self.coef == 't':
            return t
        elif self.coef == '1-t':
            return 1-t
        elif self.coef == 'cos(pi*t/2)':
            return torch.cos( 1/2 * np.pi * t)
        elif self.coef == 'sin(pi*t/2)':
            return torch.sin( 1/2 * np.pi * t)
        elif self.coef == 'sqrt(a*t*(1-t))':
            return torch.sqrt(self.a*t*(1-t))
        elif self.coef == 'sqrt(a*t*(1-t))+b':
            return torch.sqrt(self.a*t*(1-t)) + self.b
        elif self.coef == 'a*t*(1-t)':
            return self.a*t*(1-t)
        elif self.coef == 'a*sin(pi*t)':
            return self.a*torch.sin(np.pi*t)
        else:
            return torch.zeros_like(t) + float(self.coef)
        
    def dt(self, t):
        if self.coef == 't':
            return torch.ones_like(t)
        elif self.coef == '1-t':
            return -torch.ones_like(t)
        elif self.coef == 'cos(pi*t/2)':
            return -torch.sin( 1/2 * np.pi * t) * 1/2 * np.pi
        elif self.coef == 'sin(pi*t/2)':
            return torch.cos( 1/2 * np.pi * t) * 1/2 * np.pi
        elif self.coef == 'sqrt(a*t*(1-t))':
            return 1 / 2 / torch.sqrt(t*(1-t)) * self.a**0.5 * (1 - 2*t)
        elif self.coef == 'sqrt(a*t*(1-t))+b':
            return 1 / 2 / torch.sqrt(t*(1-t)) * self.a**0.5 * (1 - 2*t)
        elif self.coef == 'a*t*(1-t)':
            return self.a - 2 * t * self.a
        elif self.coef == 'a*sin(pi*t)':
            return self.a * np.pi * torch.cos(np.pi*t)
        else:
            return torch.zeros_like(t)    

    def tdt(self, t):
        if self.coef == 'sqrt(a*t*(1-t))': 
            return 1/2 * self.a * (1 - 2*t)
        else:
            return self.t(t) * self.dt(t)

def get_target(cfg):
    if cfg.target.name == "gmm":

        try: 
            if cfg.train.deepbar == True:
                from energy.gmm import GMM
                from energy.gaussian import Gaussian 
                target1 = GMM(dim=cfg.target.dim, 
                            n_mixes=cfg.target.gmm1.num_gaussian, 
                            loc_scaling=cfg.target.gmm1.loc, 
                            log_var_scaling=cfg.target.gmm1.scale, 
                            mean_shift=cfg.target.gmm1.shift, 
                            seed=cfg.target.gmm1.seed,
                            device=cfg.device)
                target2 = GMM(dim=cfg.target.dim, 
                            n_mixes=cfg.target.gmm2.num_gaussian, 
                            loc_scaling=cfg.target.gmm2.loc, 
                            log_var_scaling=cfg.target.gmm2.scale, 
                            mean_shift=cfg.target.gmm2.shift, 
                            seed=cfg.target.gmm2.seed,
                            device=cfg.device)
                # make one to Gaussian
                if cfg.train.deepbar_target == 1:
                    target2 = Gaussian(dim=cfg.target.dim, device=cfg.device)
                elif cfg.train.deepbar_target == 2:
                    target1 = Gaussian(dim=cfg.target.dim, device=cfg.device)
                print("Using GMM with deepbar")
                return target1, target2
        except:
            pass
        from energy.gmm import GMM
        target1 = GMM(dim=cfg.target.dim, 
                      n_mixes=cfg.target.gmm1.num_gaussian, 
                      loc_scaling=cfg.target.gmm1.loc, 
                      log_var_scaling=cfg.target.gmm1.scale, 
                      mean_shift=cfg.target.gmm1.shift, 
                      seed=cfg.target.gmm1.seed,
                      device=cfg.device)
        target2 = GMM(dim=cfg.target.dim, 
                      n_mixes=cfg.target.gmm2.num_gaussian, 
                      loc_scaling=cfg.target.gmm2.loc, 
                      log_var_scaling=cfg.target.gmm2.scale, 
                      mean_shift=cfg.target.gmm2.shift, 
                      seed=cfg.target.gmm2.seed,
                      device=cfg.device)
        return target1, target2
    

    elif cfg.target.name == "lj":
        from energy.lj import LennardJonesPotential
        try:
            score_path = cfg.target.lj1.score_path
        except:
            score_path = None
        try:
            rescaling = cfg.target.lj1.rescaling
        except:
            rescaling = 1
        print('Scaling:', rescaling)
        target1 = LennardJonesPotential(dim=cfg.target.dim, 
                                        n_particles=cfg.target.lj1.n_particles, 
                                        two_event_dims=False,
                                        device=cfg.device,
                                        eps=cfg.target.lj1.scale,
                                        rm=cfg.target.lj1.scale,
                                        sample_path=cfg.target.lj1.sample_path,
                                        score_path=score_path,
                                        rescaling=rescaling)
        try:
            score_path = cfg.target.lj2.score_path
        except:
            score_path = None
        try:
            rescaling = cfg.target.lj2.rescaling
        except:
            rescaling = 1
        print('Scaling:', rescaling)
        target2 = LennardJonesPotential(dim=cfg.target.dim, 
                                        n_particles=cfg.target.lj2.n_particles, 
                                        two_event_dims=False,
                                        device=cfg.device,
                                        eps=cfg.target.lj2.scale,
                                        rm=cfg.target.lj2.scale,
                                        sample_path=cfg.target.lj2.sample_path,
                                        score_path=score_path,
                                        rescaling=rescaling)
        return target1, target2
    elif cfg.target.name == "aldp":
        from energy.aldp import AldpBoltzmann
        try:
            score_path = cfg.target.aldp1.score_path
        except:
            score_path = None
        target1 = AldpBoltzmann(temperature=cfg.target.aldp1.temperature, 
                                env=cfg.target.aldp1.env, 
                                n_threads=cfg.target.aldp1.n_threads, 
                                sample_path=cfg.target.aldp1.sample_path, 
                                device=cfg.device,
                                score_path=score_path,
                                lamb=cfg.target.aldp1.lambd,
                                scaling=cfg.target.aldp1.scaling)
        try:
            score_path = cfg.target.aldp2.score_path
        except:
            score_path = None

        target2 = AldpBoltzmann(temperature=cfg.target.aldp2.temperature,
                                env=cfg.target.aldp2.env, 
                                n_threads=cfg.target.aldp2.n_threads, 
                                sample_path=cfg.target.aldp2.sample_path, 
                                device=cfg.device,
                                score_path=score_path,
                                lamb=cfg.target.aldp2.lambd,
                                scaling=cfg.target.aldp2.scaling)

        try: 
            if cfg.train.deepbar == True:
                from energy.gaussian import Gaussian_zero_center 
                # make one to Gaussian
                if cfg.train.deepbar_target == 1:
                    target2 = Gaussian_zero_center(3, 22, device=cfg.device)
                elif cfg.train.deepbar_target == 2:
                    target1 = Gaussian_zero_center(3, 22, device=cfg.device)
                print("ALDP with deepbar")
                # return target1, target2
        except:
            pass

        return target1, target2



    elif cfg.target.name == "phi4":
        from energy.phi4 import phi4
        target1 = phi4(positive=cfg.target.phi4_1.positive, 
                       sample_path=cfg.target.phi4_1.sample_path, 
                       device=cfg.device,)
        target2 = phi4(positive=cfg.target.phi4_2.positive, 
                       sample_path=cfg.target.phi4_2.sample_path, 
                       device=cfg.device,)
        return target1, target2   

def get_sampler_from_samples(cfg, x1, x2):
    if cfg.train.OT_pair:
        from torchcfm.optimal_transport import OTPlanSampler
    
    def sample(n):
        x1_shuffled, x2_shuffled = x1[torch.randperm(x1.shape[0])], x2[torch.randperm(x2.shape[0])]
        x1_subset, x2_subset = x1_shuffled[:n], x2_shuffled[:n]
        if cfg.train.OT_pair:
            sampler = OTPlanSampler(method="exact")
            x1_subset, x2_subset = sampler.sample_plan(x1_subset, x2_subset)
        return x1_subset, x2_subset
    return sample

def get_sampler_from_dataset(cfg, dataset1, dataset2):
    if cfg.train.OT_pair:
        from torchcfm.optimal_transport import OTPlanSampler
    def sample(n):
        x1, x2 = next(iter(dataset1)), next(iter(dataset2))
        return x1, x2
        # TODO: OT sampler not supported yet
    return sample
    
def get_sampler_from_target(cfg, target1, target2):
    if cfg.train.OT_pair:
        from torchcfm.optimal_transport import OTPlanSampler

    def sample(n, use_ot=True):
        try:
            ot_batch = cfg.train.OT_batch
        except: 
            ot_batch = n
        x1 = target1.sample((ot_batch, ))
        x2 = target2.sample((ot_batch, ))
        if cfg.train.OT_pair and use_ot:
            sampler = OTPlanSampler(method="exact")
            x1, x2 = sampler.sample_plan(x1, x2)
        return x1[:n], x2[:n]
    return sample

def get_sampler_with_grad_from_target(cfg, target1, target2):
    if cfg.train.OT_pair:
        from torchcfm.optimal_transport import OTPlanSampler

    def sample(n,  use_ot=True):
        try:
            ot_batch = cfg.train.OT_batch
        except: 
            ot_batch = n
        if use_ot == False:
            ot_batch = n
        x1, score1 = target1.get_sample_and_score((ot_batch, ))
        x2, score2 = target2.get_sample_and_score((ot_batch, ))
        if cfg.train.OT_pair and use_ot:
            sampler = OTPlanSampler(method="exact")

            pi = sampler.get_map(x1, x2)
            i, j = sampler.sample_map(pi, x1.shape[0])
            return x1[i][:n], x2[j][:n], score1[i][:n], score2[j][:n]
        return x1[:n], x2[:n], score1[:n], score2[:n]
    return sample


def get_sampler_with_grad_fix_order(cfg, target1, target2):
    if cfg.train.OT_pair:
        from torchcfm.optimal_transport import OTPlanSampler

    def sample(n,  use_ot=True):
        try:
            ot_batch = cfg.train.OT_batch
        except: 
            ot_batch = n
            
        idx = np.random.choice(target1.sample_data.shape[0], size=ot_batch, replace=False)
        x1, score1 = target1.get_sample_and_score_idx(idx)
        x2, score2 = target2.get_sample_and_score_idx(idx)
        if cfg.train.OT_pair and use_ot:
            sampler = OTPlanSampler(method="exact")
            pi = sampler.get_map(x1, x2)
            i, j = sampler.sample_map(pi, x1.shape[0])
            return x1[i][:n], x2[j][:n], score1[i][:n], score2[j][:n]
        return x1[:n], x2[:n], score1[:n], score2[:n]
    return sample


def get_marginal_plot_fn(cfg):
    if cfg.target.name == "gmm":
        def evaluate(x1, x2, x1_hat, x2_hat, save_path, *args, **kwargs):
            # Create subplots
            fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))

            axes[0].scatter(x1[:, 0], x1[:, 1], s=1, alpha=0.5)
            axes[0].scatter(x1_hat[:, 0], x1_hat[:, 1], s=1, alpha=0.5)
            axes[1].scatter(x2[:, 0], x2[:, 1], s=1, alpha=0.5)
            axes[1].scatter(x2_hat[:, 0], x2_hat[:, 1], s=1, alpha=0.5)
            plt.savefig(save_path)
            plt.close()
        return evaluate

    
    elif cfg.target.name == "lj":

        def evaluate(x1, x2, x1_hat, x2_hat, save_path, target1, target2, *args, **kwargs):
            
            fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5))
            
            x1_energy = target1.energy(x1).detach().cpu().numpy()
            x1_energy_hat = target1.energy(x1_hat).detach().cpu().numpy()
            x2_energy = target2.energy(x2).detach().cpu().numpy()
            x2_energy_hat = target2.energy(x2_hat).detach().cpu().numpy()
            if cfg.target.lj1.n_particles == 13:
                min1, max1 = (-100, 50) 
                min2, max2 = (-100, 50) 
            elif cfg.target.lj1.n_particles == 55:
                min1, max1 = (-400, 100) 
                min2, max2 = (-400, 100)
            else:
                min1, max1 = (0, 200) if target1._eps == 0 else (-200, 0)
                min2, max2 = (-550, -350) 
            axes[1, 0].hist(x1_energy, np.linspace(min1, max1, 100), density=1, alpha=1, histtype='step', label='x1')
            axes[1, 0].hist(x1_energy_hat, np.linspace(min1, max1, 100), density=1, alpha=1, histtype='step', label='x1 pred')
            axes[1, 1].hist(x2_energy, np.linspace(min2, max2, 100), density=1, alpha=1, histtype='step', label='x2')
            axes[1, 1].hist(x2_energy_hat, np.linspace(min2, max2, 100), density=1, alpha=1, histtype='step', label='x2 pred')
            axes[1, 0].legend()
            axes[1, 1].legend()           
            
            
            x1 = x1 / target1.scaling
            x2 = x2 / target2.scaling
            x1_hat = x1_hat / target1.scaling
            x2_hat = x2_hat / target2.scaling

            

            def get_dist(x):
                x = (((x.reshape(-1, cfg.target.lj1.n_particles, 1, cfg.target.lj1.n_dim) - x.reshape(-1, 1,cfg.target.lj1.n_particles, cfg.target.lj1.n_dim))**2).sum(-1).sqrt()).cpu()
                diagx = torch.triu_indices(x.shape[1], x.shape[1], 1)
                return x[:, diagx[0], diagx[1]].flatten()
        
            x1_dist = get_dist(x1)
            x1_hat_dist = get_dist(x1_hat)
            x2_dist = get_dist(x2)
            x2_hat_dist = get_dist(x2_hat)
            axes[0, 0].hist(x1_dist, 100, density=1, alpha=1, histtype='step', label='x1')
            axes[0, 0].hist(x1_hat_dist, 100, density=1, alpha=1, histtype='step', label='x1 pred')
            axes[0, 1].hist(x2_dist, 100, density=1, alpha=1, histtype='step', label='x2')
            axes[0, 1].hist(x2_hat_dist, 100, density=1, alpha=1, histtype='step', label='x2 pred')
            axes[0, 0].legend()
            axes[0, 1].legend()
            

            
            
            plt.savefig(save_path)
            plt.close()

        return evaluate

    elif cfg.target.name == "aldp":
        import mdtraj
        import matplotlib as mpl

        def evaluate(x1, x2, x1_hat, x2_hat, save_path, target1, target2, *args, **kwargs):
            x1 = x1 / target1.scaling
            x2 = x2 / target2.scaling
            x1_hat = x1_hat / target1.scaling
            x2_hat = x2_hat / target2.scaling
            
            # target2 = target1

            # x1 = flip_chirality(x1)
            # x2 = flip_chirality(x2)
            # x1_hat = flip_chirality(x1_hat)
            # x2_hat = flip_chirality(x2_hat)
            
            x1_np = x1.detach().cpu().numpy()
            x2_np = x2.detach().cpu().numpy()
            x1_hat_np = x1_hat.detach().cpu().numpy()
            x2_hat_np = x2_hat.detach().cpu().numpy()

            # plot x1 and x1_hat
            aldp = target1.system
            topology = mdtraj.Topology.from_openmm(aldp.topology)
            test_traj = mdtraj.Trajectory(x1_np.reshape(-1, 22, 3), topology)
            sampled_traj = mdtraj.Trajectory(x1_hat_np.reshape(-1, 22, 3), topology)
            psi_d = mdtraj.compute_psi(test_traj)[1].reshape(-1)
            phi_d = mdtraj.compute_phi(test_traj)[1].reshape(-1)
            is_nan = np.logical_or(np.isnan(psi_d), np.isnan(phi_d))
            not_nan = np.logical_not(is_nan)
            psi_d = psi_d[not_nan]
            phi_d = phi_d[not_nan]
            psi = mdtraj.compute_psi(sampled_traj)[1].reshape(-1)
            phi = mdtraj.compute_phi(sampled_traj)[1].reshape(-1)
            is_nan = np.logical_or(np.isnan(psi), np.isnan(phi))
            not_nan = np.logical_not(is_nan)
            psi = psi[not_nan]
            phi = phi[not_nan]       

            # Ramachandran plot
            plt.figure(figsize=(20, 20))
            plt.subplot(2, 2, 1)

            plt.hist2d(phi_d, psi_d, bins=64, norm=mpl.colors.LogNorm(),
                        range=[[-np.pi, np.pi], [-np.pi, np.pi]])
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('$\phi$', fontsize=24)
            plt.ylabel('$\psi$', fontsize=24)

            plt.subplot(2, 2, 2)
            plt.hist2d(phi, psi, bins=64, norm=mpl.colors.LogNorm(),
                        range=[[-np.pi, np.pi], [-np.pi, np.pi]])
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('$\phi$', fontsize=24)
            plt.ylabel('$\psi$', fontsize=24)


            # plot x2 and x2_hat
            aldp = target1.system
            topology = mdtraj.Topology.from_openmm(aldp.topology)
            test_traj = mdtraj.Trajectory(x2_np.reshape(-1, 22, 3), topology)
            sampled_traj = mdtraj.Trajectory(x2_hat_np.reshape(-1, 22, 3), topology)
            psi_d = mdtraj.compute_psi(test_traj)[1].reshape(-1)
            phi_d = mdtraj.compute_phi(test_traj)[1].reshape(-1)
            is_nan = np.logical_or(np.isnan(psi_d), np.isnan(phi_d))
            not_nan = np.logical_not(is_nan)
            psi_d = psi_d[not_nan]
            phi_d = phi_d[not_nan]
            psi = mdtraj.compute_psi(sampled_traj)[1].reshape(-1)
            phi = mdtraj.compute_phi(sampled_traj)[1].reshape(-1)
            is_nan = np.logical_or(np.isnan(psi), np.isnan(phi))
            not_nan = np.logical_not(is_nan)
            psi = psi[not_nan]
            phi = phi[not_nan]  

            plt.subplot(2, 2, 3)

            plt.hist2d(phi_d, psi_d, bins=64, norm=mpl.colors.LogNorm(),
                        range=[[-np.pi, np.pi], [-np.pi, np.pi]])
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('$\phi$', fontsize=24)
            plt.ylabel('$\psi$', fontsize=24)

            plt.subplot(2, 2, 4)
            plt.hist2d(phi, psi, bins=64, norm=mpl.colors.LogNorm(),
                        range=[[-np.pi, np.pi], [-np.pi, np.pi]])
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('$\phi$', fontsize=24)
            plt.ylabel('$\psi$', fontsize=24)

            plt.savefig(save_path, dpi=300)
            plt.close()


        return evaluate
    if cfg.target.name == "phi4":
        def evaluate(x1, x2, x1_hat, x2_hat, save_path, *args, **kwargs):
            # Create subplots
            fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))

            axes[0].hist(x1.mean(-1), bins=100, density=True, alpha=0.5, histtype='step')
            axes[0].hist(x1_hat.mean(-1), bins=100, density=True, alpha=0.5, histtype='step')
            axes[1].hist(x2.mean(-1), bins=100, density=True, alpha=0.5, histtype='step')
            axes[1].hist(x2_hat.mean(-1), bins=100, density=True, alpha=0.5, histtype='step')
            plt.savefig(save_path)
            plt.close()
        return evaluate 


def detect_chirality(coords):
    """
    Detects D-residues in a batch of ALDP molecules.
    Input: coords of shape (N, 66) (N samples, 22 atoms × 3D)
    Output: Boolean mask of shape (N,) where True means D-configuration.
    """
    coords = coords.view(-1, 22, 3)  # (N, 22, 3)

    # ALDP atom indices
    n, ca, c, cb = 6, 8, 14, 10  

    # Compute vectors
    vec_n = coords[:, n, :] - coords[:, ca, :]  # N - CA
    vec_c = coords[:, c, :] - coords[:, ca, :]  # C - CA
    vec_cb = coords[:, cb, :] - coords[:, ca, :]  # CB - CA

    # **Normalization Fix**: Normalize vectors to avoid scale issues
    vec_n = vec_n / vec_n.norm(dim=1, keepdim=True)
    vec_c = vec_c / vec_c.norm(dim=1, keepdim=True)
    vec_cb = vec_cb / vec_cb.norm(dim=1, keepdim=True)

    # Compute chirality determinant
    chirality = torch.einsum('ij,ij->i', torch.cross(vec_n, vec_c, dim=1), vec_cb)

    # **Fix Chirality Check**: Negative means D-form
    d_mask = chirality < 0  
    return d_mask  # Shape: (N,)

def flip_d_residues(coords, d_mask):
    """
    Flips only the D-residues in the batch.
    Input:
      - coords: (N, 66) tensor of atomic positions
      - d_mask: (N,) boolean tensor indicating which rows are D-configuration
    Output: Flipped tensor with only D-residues corrected.
    """
    coords = coords.view(-1, 22, 3)  # (N, 22, 3)

    # Get Cα (8) and Cβ (10) positions
    ca_pos = coords[:, 8, :]  # (N, 3)
    cb_pos = coords[:, 10, :]  # (N, 3)

    # **Fix Shape Issue**: Ensure we modify CB only when there are D-residues
    if d_mask.any():
        cb_pos[d_mask] = 2 * ca_pos[d_mask] - cb_pos[d_mask]  # Mirror CB across CA

    # Store the corrected coordinates
    coords[:, 10, :] = cb_pos  # Update CB positions
    return coords.view(-1, 66)  # Flatten back to original shape

def flip_chirality(input_tensor):
    """
    Detects and fixes chirality issues in a batch of ALDP molecules.
    Input: Tensor of shape (N, 66)
    Output: Corrected tensor of shape (N, 66)
    """
    d_mask = detect_chirality(input_tensor)  # Find D-residues
    corrected_coords = flip_d_residues(input_tensor, d_mask)  # Flip only D-configurations
    return corrected_coords  # Shape: (N, 66)
