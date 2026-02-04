"""
Training script for diffusion model using Denoising Score Matching (DSM).

This script trains a diffusion model for molecular dynamics systems using
an EGNN-based denoising network with exponential moving average (EMA).

Example usage:
    # Basic training with required arguments
    python train_dm.py --data_path data/4A_1.0_align_ot.pt --n_particles 43
    
    # Full example with all parameters (matching notebook defaults)
    python train_dm.py \\
        --data_path data/4A_1.0_align_ot.pt \\
        --n_particles 43 \\
        --data_scaling 5.0 \\
        --n_epochs 100000 \\
        --batch_size 20 \\
        --lr 1e-4 \\
        --ema_decay 0.99 \\
        --eval_freq 1000 \\
        --save_dir ./checkpoints \\
        --device cuda
    
    # Resume training from checkpoint
    python train_dm.py \\
        --data_path data/4A_1.0_align_ot.pt \\
        --n_particles 43 \\
        --resume checkpoints/ema_net_epoch_50000.pt
"""

import argparse
import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from networks.egnn import EGNN_dynamics_AD4, remove_mean


def dsm_loss(x_t, model, x0, t):
    """
    Denoising Score Matching (DSM) loss function.
    
    Parameters
    ----------
    x_t : torch.Tensor
        Noisy samples at time t, shape (batch_size, n_particles * n_dim)
    model : torch.nn.Module
        Denoising network model
    x0 : torch.Tensor
        Clean samples, shape (batch_size, n_particles * n_dim)
    t : torch.Tensor
        Time values, shape (batch_size,)
    
    Returns
    -------
    loss : torch.Tensor
        Scalar loss value
    """
    w = (model.data_sigma * t) ** 2 / (model.data_sigma ** 2 + t ** 2)
    x_hat = model(x_t, t)
    dsm_loss = (((x0 - x_hat) ** 2).sum(-1) / w).mean()
    return dsm_loss


def update_ema_model(ema_decay, online_model, ema_model):
    """
    Update exponential moving average (EMA) model parameters.
    
    Parameters
    ----------
    ema_decay : float
        EMA decay factor (typically 0.99-0.9999)
    online_model : torch.nn.Module
        Online model (being trained)
    ema_model : torch.nn.Module
        EMA model (shadow model)
    """
    with torch.no_grad():
        online_params = OrderedDict(online_model.named_parameters())
        ema_params = OrderedDict(ema_model.named_parameters())

        # Check if both models contain the same set of keys
        assert online_params.keys() == ema_params.keys()

        for name, param in online_params.items():
            # EMA update: shadow_variable -= (1 - decay) * (shadow_variable - variable)
            ema_params[name].sub_((1. - ema_decay) * (ema_params[name] - param))
            ema_params[name].requires_grad = False

        # Update buffers (e.g., batch norm statistics)
        online_buffers = OrderedDict(online_model.named_buffers())
        ema_buffers = OrderedDict(ema_model.named_buffers())

        assert online_buffers.keys() == ema_buffers.keys()

        for name, buffer in online_buffers.items():
            ema_buffers[name].copy_(buffer)


def em_solve(model, start_samples, ts, n_particles):
    """
    Euler-Maruyama solver for sampling from the diffusion model.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained denoising model (typically EMA model)
    start_samples : torch.Tensor
        Initial samples (noise), shape (batch_size, n_particles * n_dim)
    ts : torch.Tensor or np.ndarray
        Time steps for reverse diffusion, shape (n_steps,)
    n_particles : int
        Number of particles in the system
    
    Returns
    -------
    samples : torch.Tensor
        Generated samples, shape (batch_size, n_particles * n_dim)
    """
    with torch.no_grad():
        samples = start_samples.clone()
        for i in range(len(ts) - 1, 0, -1):
            t = torch.ones(samples.shape[0], 1, device=samples.device) * ts[i]
            t_prev = torch.ones(samples.shape[0], 1, device=samples.device) * ts[i - 1]

            delta_t = (t - t_prev).abs()
            x_hat = model(samples, t.squeeze(-1))
            std = torch.sqrt(2 * delta_t * t)
            score = -(samples - x_hat) / t ** 2

            dx = score * 2 * t * delta_t + std * remove_mean(
                torch.randn_like(samples), n_particles, 3
            )

            samples = samples + dx
        return samples


def get_distance_distribution(x, n_particles):
    """
    Compute pairwise distance distribution for evaluation.
    
    Parameters
    ----------
    x : torch.Tensor
        Samples, shape (batch_size, n_particles * n_dim)
    n_particles : int
        Number of particles
    
    Returns
    -------
    distances : torch.Tensor
        Flattened pairwise distances
    """
    x = x.reshape(-1, n_particles, 1, 3)
    distances = ((x - x.transpose(1, 2)) ** 2).sum(-1).sqrt()
    diag_indices = torch.triu_indices(n_particles, n_particles, 1)
    return distances[:, diag_indices[0], diag_indices[1]].flatten()


def evaluate_model(model, data, target, n_particles, device, n_samples=100, 
                   tmax=20.0, tmin=1e-6, rho=7.0, steps=200):
    """
    Evaluate the trained model by generating samples and comparing with data.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model (typically EMA model)
    data : torch.Tensor
        Training data samples
    target : object
        Target distribution with log_prob method
    n_particles : int
        Number of particles
    device : str
        Device to use ('cuda' or 'cpu')
    n_samples : int
        Number of samples to generate
    tmax : float
        Maximum time for reverse diffusion
    tmin : float
        Minimum time for reverse diffusion
    rho : float
        Time schedule parameter
    steps : int
        Number of reverse diffusion steps
    
    Returns
    -------
    samples : torch.Tensor
        Generated samples
    """
    # Set up time schedule
    ts = tmin ** (1 / rho) + np.arange(steps) / (steps - 1) * (
        tmax ** (1 / rho) - tmin ** (1 / rho)
    )
    ts = ts ** rho
    ts = torch.tensor(ts, device=device)

    with torch.no_grad():
        # Generate samples
        start_samples = remove_mean(
            torch.randn(n_samples, 3 * n_particles, device=device) * ts[-1],
            n_particles, 3
        )
        samples = em_solve(model, start_samples, ts, n_particles)

    return samples


def main():
    """
    Main training function.
    
    Example command:
        python train_dm.py --data_path data/4A_1.0_align_ot.pt --n_particles 43
    """
    parser = argparse.ArgumentParser(
        description='Train diffusion model with DSM loss',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_dm.py --data_path data/4A_1.0_align_ot.pt --n_particles 43
  
  # Full training with custom parameters
  python train_dm.py --data_path data/4A_1.0_align_ot.pt --n_particles 43 \\
      --n_epochs 100000 --batch_size 20 --lr 1e-4 --save_dir ./checkpoints
  
  # Resume from checkpoint
  python train_dm.py --data_path data/4A_1.0_align_ot.pt --n_particles 43 \\
      --resume checkpoints/ema_net_epoch_50000.pt
        """
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data file (.pt)')
    parser.add_argument('--n_particles', type=int, required=True,
                        help='Number of particles in the system')
    parser.add_argument('--data_scaling', type=float, default=5.0,
                        help='Scaling factor for data (default: 5.0)')
    
    # Model arguments
    parser.add_argument('--hidden_nf', type=int, default=256,
                        help='Hidden dimension size (default: 256)')
    parser.add_argument('--n_layers', type=int, default=5,
                        help='Number of EGNN layers (default: 5)')
    parser.add_argument('--recurrent', action='store_true', default=True,
                        help='Use recurrent EGNN layers')
    parser.add_argument('--attention', action='store_true', default=True,
                        help='Use attention in EGNN')
    parser.add_argument('--tanh', action='store_true', default=True,
                        help='Use tanh activation')
    
    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=100000,
                        help='Number of training epochs (default: 100000)')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Batch size (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--ema_decay', type=float, default=0.99,
                        help='EMA decay factor (default: 0.99)')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Gradient clipping norm (default: 5.0)')
    
    # Evaluation arguments
    parser.add_argument('--eval_freq', type=int, default=1000,
                        help='Evaluation frequency in epochs (default: 1000)')
    parser.add_argument('--n_eval_samples', type=int, default=100,
                        help='Number of samples for evaluation (default: 100)')
    parser.add_argument('--save_freq', type=int, default=1000,
                        help='Model save frequency in epochs (default: 1000)')
    
    # Sampling/solver arguments
    parser.add_argument('--tmax', type=float, default=20.0,
                        help='Maximum time for reverse diffusion (default: 20.0)')
    parser.add_argument('--tmin', type=float, default=1e-3,
                        help='Minimum time for reverse diffusion (default: 1e-6)')
    parser.add_argument('--rho', type=float, default=7.0,
                        help='Time schedule parameter rho (default: 7.0)')
    parser.add_argument('--n_steps', type=int, default=200,
                        help='Number of reverse diffusion steps (default: 200)')
    
    # Time sampling arguments
    parser.add_argument('--logt_std', type=float, default=1.2,
                        help='Standard deviation for log-time sampling (default: 1.2)')
    parser.add_argument('--logt_mean', type=float, default=-1.2,
                        help='Mean for log-time sampling (default: -1.2)')
    
    # Path arguments
    parser.add_argument('--save_dir', type=str, default='./',
                        help='Directory to save models and plots (default: ./)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (default: None)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f'Using device: {device}')
    
    # Load data
    print(f'Loading data from {args.data_path}...')
    data = torch.load(args.data_path) * args.data_scaling
    data_std = data.std().item()
    print(f'Data shape: {data.shape}, std: {data_std:.4f}')
    
    # Initialize model
    print('Initializing model...')
    denoising_net = EGNN_dynamics_AD4(
        n_particles=args.n_particles,
        n_dimension=3,
        hidden_nf=args.hidden_nf,
        device=device,
        act_fn=nn.SiLU(),
        n_layers=args.n_layers,
        recurrent=args.recurrent,
        attention=args.attention,
        condition_time=True,
        tanh=args.tanh,
        mode='egnn_dynamics',
        agg='sum',
        data_sigma=data_std
    ).to(device)
    
    # Initialize EMA model
    ema_net = deepcopy(denoising_net).to(device)
    
    # Resume from checkpoint if provided
    if args.resume and os.path.exists(args.resume):
        print(f'Loading checkpoint from {args.resume}...')
        denoising_net.load_state_dict(torch.load(args.resume, map_location=device))
        ema_net.load_state_dict(torch.load(args.resume, map_location=device))
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(denoising_net.parameters(), lr=args.lr)
    
    # Training loop
    print(f'Starting training for {args.n_epochs} epochs...')
    losses = []
    
    for epoch in tqdm(range(args.n_epochs)):
        # Sample batch
        batch_indices = torch.randint(0, data.shape[0], (args.batch_size,))
        x = remove_mean(data[batch_indices].to(device), args.n_particles, 3)
        
        # Sample time (log-space)
        logt = torch.randn(args.batch_size, device=device) * args.logt_std + args.logt_mean
        t = logt.exp()
        
        # Sample noise
        noises = remove_mean(torch.randn_like(x), args.n_particles, 3)
        
        # Create noisy samples
        x_t = remove_mean(x + noises * t[:, None], args.n_particles, 3)
        
        # Compute loss
        loss = dsm_loss(x_t, denoising_net, x, t)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoising_net.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        
        # Update EMA model
        update_ema_model(args.ema_decay, denoising_net, ema_net)
        
        # Evaluation and saving
        if (epoch + 1) % args.eval_freq == 0 or epoch == 0:
            print(f'\nEpoch {epoch + 1}/{args.n_epochs}, Loss: {loss.item():.6f}')
            
            # Generate samples for evaluation
            samples = evaluate_model(
                ema_net, data, None, args.n_particles, device,
                n_samples=args.n_eval_samples,
                tmax=args.tmax, tmin=args.tmin, rho=args.rho, steps=args.n_steps
            )
            
            # Save model
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                model_path = os.path.join(args.save_dir, f'ema_net_epoch_{epoch+1}.pt')
                torch.save(ema_net.state_dict(), model_path)
                print(f'Saved model to {model_path}')
                
                # Save loss plot
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(5, 3))
                    plt.plot(losses)
                    plt.yscale('log')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Training Loss')
                    plt.tight_layout()
                    loss_plot_path = os.path.join(args.save_dir, f'loss_plot_epoch_{epoch+1}.png')
                    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f'Saved loss plot to {loss_plot_path}')
                except ImportError:
                    print('Matplotlib not available, skipping plot generation')
    
    print('Training completed!')
    
    # Final save
    if args.save_dir:
        final_model_path = os.path.join(args.save_dir, 'ema_net_final.pt')
        torch.save(ema_net.state_dict(), final_model_path)
        print(f'Saved final model to {final_model_path}')


if __name__ == '__main__':
    main()
