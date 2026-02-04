import argparse
import torch
import numpy as np


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align configurations to a reference structure')
    parser.add_argument('--ref_conf_file', type=str, required=True,
                        help='Path to reference configuration file')
    parser.add_argument('--ref_idx', type=int, default=0,
                        help='Index of reference configuration in the file (default: 0)')
    parser.add_argument('--align_file_path', type=str, required=True,
                        help='Path to save the aligned configurations')
    parser.add_argument('--target_file', type=str, required=True,
                        help='Path to target configuration file to align')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Load reference configuration
    data = torch.load(args.ref_conf_file).to(args.device).reshape(-1, 66)
    ref = data[args.ref_idx].reshape(22, 3)

    # Load target configurations
    data = torch.load(args.target_file).to(args.device)
    b = data.reshape(-1, 22, 3).clone()
    align = superimpose_B_onto_A(ref, b).reshape(-1, 22*3)
    torch.save(align, args.align_file_path)