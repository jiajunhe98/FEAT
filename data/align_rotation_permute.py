import argparse
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def center_of_mass_batch(positions, masses=None):
    """Compute the center of mass for a batch of positions."""
    if masses is None:
        return positions.mean(dim=1, keepdim=True)  # (B, 1, 3)
    return (positions * masses[:, :, None]).sum(dim=1, keepdim=True) / masses.sum(dim=1, keepdim=True)


def kabsch_alignment_batch(A, B):
    """
    Batched Kabsch algorithm to find the optimal rotation that aligns B onto A.
    A: (N, 3) reference coordinates
    B: (B, N, 3) batch of target coordinates
    Returns: (B, 3, 3) optimal rotation matrices
    """
    # Compute covariance matrix H per batch
    H = torch.einsum("ni,bnj->bij", A, B)  # (B, 3, 3)

    # Singular Value Decomposition (SVD)
    U, _, Vt = torch.linalg.svd(H)  # U, Vt: (B, 3, 3)

    # Ensure right-handed coordinate system
    det_sign = torch.det(U @ Vt)  # (B,)
    Vt[det_sign < 0, -1, :] *= -1  # Flip last row of Vt when determinant is negative

    # Compute optimal rotation matrices
    R = U @ Vt  # (B, 3, 3)

    return R


def superimpose_B_onto_A(A, B, masses=None):
    """
    Superimposes each system in B onto the reference system A.
    A: (N, 3) reference coordinates
    B: (B, N, 3) batch of target coordinates
    masses: Optional (B, N) mass array for weighted alignment
    Returns: (B, N, 3) aligned B coordinates
    """
    # Compute centers of mass
    A_com = A.mean(dim=0, keepdim=True)  # (1, 3)
    B_com = center_of_mass_batch(B, masses)  # (B, 1, 3)

    # Center the structures
    A_centered = A - A_com  # (N, 3)
    B_centered = B - B_com  # (B, N, 3)

    # Compute optimal rotation
    R = kabsch_alignment_batch(A_centered, B_centered)  # (B, 3, 3)

    # Apply rotation and translation
    B_aligned = torch.einsum("bij,bnj->bni", R, B_centered) + A_com  # (B, N, 3)

    return B_aligned


def align_and_permute(X, Y):
    """
    Aligns two point sets (X and Y) using SVD (Procrustes analysis),
    then finds the optimal permutation to minimize RMSD.
    
    Parameters:
    X: (N, 3) numpy array - First structure
    Y: (N, 3) numpy array - Second structure
    
    Returns:
    row_ind: (N,) numpy array - Row indices
    col_ind: (N,) numpy array - Permutation indices for Y
    """
    # Compute the pairwise squared distances
    cost_matrix = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2) ** 2
    
    # Solve the assignment problem using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    return row_ind, col_ind


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align configurations with rotation and permutation')
    parser.add_argument('--ref_conf_file', type=str, required=True,
                        help='Path to reference configuration file')
    parser.add_argument('--ref_idx', type=int, default=0,
                        help='Index of reference configuration in the file (default: 0)')
    parser.add_argument('--target_file', type=str, required=True,
                        help='Path to target configuration file to align')
    parser.add_argument('--align_file_path', type=str, required=True,
                        help='Path to save the aligned configurations')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of iterations of alignment and permutation (default: 5)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Load reference configuration
    ref = torch.load(args.ref_conf_file).to(args.device)
    a = ref[args.ref_idx].reshape(-1, 3).clone()
    n_atoms = a.shape[0]
    a = a - a.mean(0, True)
    
    # Load target configurations
    target_data = torch.load(args.target_file).to(args.device)
    b = target_data.reshape(-1, n_atoms, 3).clone()
    b = b - b.mean(1, True)
    
    lj_new = b
    
    # Iterative alignment and permutation
    for _ in range(args.iterations):
        lj_new = superimpose_B_onto_A(a, lj_new)
        
        for i in tqdm(range(lj_new.shape[0]), desc=f'Permutation iteration {_+1}/{args.iterations}'):
            row_ind, col_ind = align_and_permute(a.cpu().numpy(), lj_new[i].cpu().clone().numpy())
            lj_new[i] = lj_new[i].clone()[col_ind]
    
    torch.save(lj_new.reshape(-1, n_atoms * 3), args.align_file_path)
