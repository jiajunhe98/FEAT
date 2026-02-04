"""
Inference script for diffusion model using Jarzynski equality for free energy estimation.

This script performs inference using a trained diffusion model to estimate free energy
differences using both forward and reverse paths with the Jarzynski equality.

Example usage:
    # Basic inference
    python inference_dm.py --model_path checkpoints/ema_net_final.pt --n_particles 43
    
    # Full inference with custom parameters
    python inference_dm.py \\
        --model_path checkpoints/ema_net_final.pt \\
        --n_particles 43 \\
        --data_path data/4A_0.0_align_ot.pt \\
        --data_scaling 5.0 \\
        --n_samples 1000 \\
        --n_batches 10 \\
        --tmax 100 --tmin 1e-3 --rho 7 --steps 1000 \\
        --output_dir ./results
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from networks.egnn import EGNN_dynamics_AD4, remove_mean


def em_solve_reverse(model, start_samples, ts, n_particles, verbose=False):
    """
    Euler-Maruyama solver for reverse diffusion (backward path).
    Computes work along the path for Jarzynski equality.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained denoising model
    start_samples : torch.Tensor
        Initial samples (noise), shape (batch_size, n_particles * n_dim)
    ts : torch.Tensor or np.ndarray
        Time steps for reverse diffusion, shape (n_steps,)
    n_particles : int
        Number of particles in the system
    verbose : bool
        Whether to print forward/backward log probabilities
    
    Returns
    -------
    samples : torch.Tensor
        Generated samples, shape (batch_size, n_particles * n_dim)
    W : torch.Tensor
        Work values (Jarzynski work), shape (batch_size,)
    all_samples : list
        List of all samples along the path
    """
    W = torch.zeros(start_samples.shape[0], device=start_samples.device)
    all_samples = []
    
    with torch.no_grad():
        samples = remove_mean(start_samples, n_particles, 3)
        all_samples.append(samples)
        
        for i in range(len(ts) - 1, 0, -1):
            t = torch.ones(samples.shape[0], 1, device=samples.device) * ts[i]
            t_prev = torch.ones(samples.shape[0], 1, device=samples.device) * ts[i - 1]

            delta_t = (t - t_prev).abs()
            x_hat = remove_mean(model(samples, t.squeeze(-1)), n_particles, 3)

            # Mean and std for forward transition
            mean = t_prev ** 2 / t ** 2 * samples + (1 - t_prev ** 2 / t ** 2) * x_hat
            std = torch.sqrt(t_prev ** 2 / t ** 2 * (t ** 2 - t_prev ** 2))

            new_samples = mean + std * remove_mean(torch.randn_like(samples), n_particles, 3)

            # Forward log probability
            fwd = torch.distributions.Normal(mean, std).log_prob(new_samples)
            fwd = fwd.sum(-1)

            # Backward log probability
            new_mean = new_samples
            new_std = torch.sqrt(t ** 2 - t_prev ** 2)
            bwd = torch.distributions.Normal(new_mean, new_std).log_prob(samples)
            bwd = bwd.sum(-1)

            # Accumulate work
            W = W - fwd + bwd
            
            if verbose:
                print(f"Step {i}: fwd={fwd.mean().item():.4f}, bwd={bwd.mean().item():.4f}")
            
            all_samples.append(new_samples)
            samples = new_samples
            
    return samples, W, all_samples


def em_solve_forward(model, start_samples, ts, n_particles, verbose=False):
    """
    Euler-Maruyama solver for forward diffusion (forward path).
    Computes work along the path for Jarzynski equality.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained denoising model
    start_samples : torch.Tensor
        Initial samples from data, shape (batch_size, n_particles * n_dim)
    ts : torch.Tensor or np.ndarray
        Time steps for forward diffusion, shape (n_steps,)
    n_particles : int
        Number of particles in the system
    verbose : bool
        Whether to print forward/backward log probabilities
    
    Returns
    -------
    samples : torch.Tensor
        Final noisy samples, shape (batch_size, n_particles * n_dim)
    W : torch.Tensor
        Work values (Jarzynski work), shape (batch_size,)
    """
    W = torch.zeros(start_samples.shape[0], device=start_samples.device)
    
    with torch.no_grad():
        samples = remove_mean(start_samples, n_particles, 3)
        
        for i in range(len(ts) - 1):
            t = torch.ones(samples.shape[0], 1, device=samples.device) * ts[i]
            t_next = torch.ones(samples.shape[0], 1, device=samples.device) * ts[i + 1]

            # Forward transition
            std = torch.sqrt(t_next ** 2 - t ** 2)
            mean = samples
            new_samples = mean + std * remove_mean(torch.randn_like(samples), n_particles, 3)

            # Forward log probability
            fwd = torch.distributions.Normal(mean, std).log_prob(new_samples)
            fwd = fwd.sum(-1)

            # Backward transition
            x_hat = remove_mean(model(new_samples, t_next.squeeze(-1)), n_particles, 3)
            new_mean = t ** 2 / t_next ** 2 * new_samples + (1 - t ** 2 / t_next ** 2) * x_hat
            new_std = torch.sqrt(t ** 2 / t_next ** 2 * (t_next ** 2 - t ** 2))
            
            # Backward log probability
            bwd = torch.distributions.Normal(new_mean, new_std).log_prob(samples)
            bwd = bwd.sum(-1)

            # Accumulate work
            W = W - fwd + bwd
            
            if verbose:
                print(f"Step {i}: fwd={fwd.mean().item():.4f}, bwd={bwd.mean().item():.4f}")

            samples = new_samples
            
    return samples, W


def compute_free_energy_difference(W1, W2, n_iter=10000, tol=1e-4):
    """
    Compute free energy difference using Jarzynski equality and Bennett acceptance ratio.
    
    Parameters
    ----------
    W1 : torch.Tensor
        Work values from reverse path
    W2 : torch.Tensor
        Work values from forward path
    n_iter : int
        Maximum number of iterations for Bennett acceptance ratio
    tol : float
        Convergence tolerance
    
    Returns
    -------
    df1 : float
        Free energy from reverse path (logsumexp)
    df2 : float
        Free energy from forward path (logsumexp)
    df_mean : float
        Mean of df1 and df2
    df_bar : float
        Bennett acceptance ratio estimate
    """
    # Simple Jarzynski estimates
    df1 = torch.logsumexp(W1, 0).item() - np.log(W1.shape[0])
    df2 = -torch.logsumexp(W2, 0).item() + np.log(W2.shape[0])
    df_mean = (df1 + df2) / 2
    
    # Bennett acceptance ratio (BAR)
    df_bar = df_mean
    for _ in range(n_iter):
        df_bar_new = (
            torch.logsumexp(torch.nn.LogSigmoid()(W1 - df_bar), 0) -
            torch.logsumexp(torch.nn.LogSigmoid()(W2 + df_bar), 0) + df_bar
        ).item()
        
        if np.abs(df_bar_new - df_bar) < tol:
            df_bar = df_bar_new
            break
        df_bar = df_bar_new
    
    return df1, df2, df_mean, df_bar


def main():
    """
    Main inference function.
    
    Example command:
        python inference_dm.py --model_path checkpoints/ema_net_final.pt --n_particles 43
    """
    parser = argparse.ArgumentParser(
        description='Inference with diffusion model using Jarzynski equality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python inference_dm.py --model_path checkpoints/ema_net_final.pt --n_particles 43
  
  # Full inference with data
  python inference_dm.py --model_path checkpoints/ema_net_final.pt --n_particles 43 \\
      --data_path data/4A_0.0_align_ot.pt --data_scaling 5.0 --n_samples 1000
  
  # Custom sampling parameters
  python inference_dm.py --model_path checkpoints/ema_net_final.pt --n_particles 43 \\
      --tmax 100 --tmin 1e-3 --rho 7 --steps 1000
        """
    )
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--n_particles', type=int, required=True,
                        help='Number of particles in the system')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data file for forward path (optional)')
    parser.add_argument('--data_scaling', type=float, default=5.0,
                        help='Scaling factor for data (default: 5.0)')
    
    # Model arguments (must match training)
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
    
    # Sampling arguments
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples per batch (default: 100)')
    parser.add_argument('--n_batches', type=int, default=10,
                        help='Number of batches to run (default: 10)')
    
    # Solver arguments (matching notebook defaults)
    parser.add_argument('--tmax', type=float, default=100.0,
                        help='Maximum time for reverse diffusion (default: 100.0)')
    parser.add_argument('--tmin', type=float, default=1e-3,
                        help='Minimum time for reverse diffusion (default: 1e-3)')
    parser.add_argument('--rho', type=float, default=7.0,
                        help='Time schedule parameter rho (default: 7.0)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of reverse diffusion steps (default: 1000)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Directory to save results (default: ./)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress during sampling')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f'Using device: {device}')
    
    # Set up time schedule
    ts = args.tmin ** (1 / args.rho) + np.arange(args.steps) / (args.steps - 1) * (
        args.tmax ** (1 / args.rho) - args.tmin ** (1 / args.rho)
    )
    ts = ts ** args.rho
    ts = torch.tensor(ts, device=device)
    print(f'Time schedule: tmin={args.tmin:.2e}, tmax={args.tmax:.2f}, steps={args.steps}')
    
    # Load data if provided (for forward path)
    data = None
    if args.data_path and os.path.exists(args.data_path):
        print(f'Loading data from {args.data_path}...')
        data = torch.load(args.data_path) * args.data_scaling
        print(f'Data shape: {data.shape}')
    
    # Initialize model (architecture must match training)
    print('Initializing model...')
    # We need data_std for model initialization, use a default or load from checkpoint
    # For inference, we'll use a placeholder and load the actual weights
    data_std = 1.0  # Will be overridden by loaded model weights
    
    model = EGNN_dynamics_AD4(
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
    
    # Load model weights
    print(f'Loading model from {args.model_path}...')
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    model.requires_grad_(False)
    
    # Get actual data_std from model if available
    if hasattr(model, 'data_sigma'):
        print(f'Model data_sigma: {model.data_sigma:.4f}')
    
    print(f'Model loaded successfully. Running inference...')
    
    # Reverse path: sample from noise
    print('\n=== Reverse Path (Noise -> Data) ===')
    W1s = []
    
    for batch_idx in range(args.n_batches):
        print(f'Batch {batch_idx + 1}/{args.n_batches}...')
        start_samples = remove_mean(
            torch.randn(args.n_samples, 3 * args.n_particles, device=device) * ts[-1],
            args.n_particles, 3
        )
        
        samples, W, _ = em_solve_reverse(
            model, start_samples.clone(), ts, args.n_particles, verbose=args.verbose
        )
        
        # Adjust work for initial noise distribution
        W -= torch.distributions.Normal(0, ts[-1]).log_prob(start_samples).sum(-1)
        
        # Note: For full free energy, you would add target.log_prob(samples) here
        # but that requires a target distribution object
        # W += target.log_prob(samples)
        
        W1s.append(W.flatten())
    
    W1s = torch.cat(W1s, 0)
    print(f'Reverse path: {W1s.shape[0]} work values collected')
    print(f'  Mean: {W1s.mean().item():.4f}, Std: {W1s.std().item():.4f}')
    
    # Forward path: sample from data (if available)
    W2s = []
    if data is not None:
        print('\n=== Forward Path (Data -> Noise) ===')
        
        for batch_idx in range(args.n_batches):
            print(f'Batch {batch_idx + 1}/{args.n_batches}...')
            # Sample random data points
            indices = np.random.choice(data.shape[0], args.n_samples, replace=False)
            start_samples = remove_mean(
                data[indices].to(device), args.n_particles, 3
            )
            
            samples, W = em_solve_forward(
                model, start_samples.clone(), ts, args.n_particles, verbose=args.verbose
            )
            
            # Adjust work for final noise distribution
            W += torch.distributions.Normal(0, ts[-1]).log_prob(samples).sum(-1)
            
            # Note: For full free energy, you would subtract target.log_prob(start_samples) here
            # W -= target.log_prob(start_samples)
            
            W2s.append(W.flatten())
        
        W2s = torch.cat(W2s, 0)
        print(f'Forward path: {W2s.shape[0]} work values collected')
        print(f'  Mean: {W2s.mean().item():.4f}, Std: {W2s.std().item():.4f}')
        
        # Compute free energy differences
        print('\n=== Free Energy Estimates ===')
        df1, df2, df_mean, df_bar = compute_free_energy_difference(W1s, W2s)
        
        print(f'Reverse path (DF1): {df1:.4f}')
        print(f'Forward path (DF2): {df2:.4f}')
        print(f'Mean (DF1 + DF2)/2: {df_mean:.4f}')
        print(f'Bennett acceptance ratio (BAR): {df_bar:.4f}')
        
        # Save results
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            results = {
                'W1': W1s.cpu().numpy(),
                'W2': W2s.cpu().numpy(),
                'df1': df1,
                'df2': df2,
                'df_mean': df_mean,
                'df_bar': df_bar,
                'n_samples': W1s.shape[0],
                'tmax': args.tmax,
                'tmin': args.tmin,
                'rho': args.rho,
                'steps': args.steps,
            }
            results_path = os.path.join(args.output_dir, 'free_energy_results.npz')
            np.savez(results_path, **results)
            print(f'\nResults saved to {results_path}')
    else:
        print('\nNote: Forward path skipped (no data provided)')
        print(f'Reverse path work values: {W1s.shape[0]} samples')
        print(f'  Mean: {W1s.mean().item():.4f}, Std: {W1s.std().item():.4f}')
        
        # Save reverse path only
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            results_path = os.path.join(args.output_dir, 'reverse_path_work.npz')
            np.savez(results_path, W1=W1s.cpu().numpy())
            print(f'Results saved to {results_path}')
    
    print('\nInference completed!')


if __name__ == '__main__':
    main()
