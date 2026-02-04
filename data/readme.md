# Data Folder

Place your data files in this folder.

## Alignment for Optimal Transport

For larger systems (ALDP, LJ-128, or larger), it is beneficial to use mini-batch optimal transport (OT) pairs when training the transport model. However, due to the rotation-invariance (and sometimes permutation-invariance) of these systems, naive OT with Euclidean distance typically fails to provide sufficient signal for training.

Therefore, we recommend first aligning your configurations using the alignment scripts in this folder:

- **`align_rotation.py`**: Aligns configurations using rotation only (Kabsch algorithm)
- **`align_rotation_permute.py`**: Aligns configurations using both rotation and permutation (for systems with indistinguishable particles)

Both scripts accept command-line arguments for reference configuration file, reference index, target file, output path, and (for the permutation version) number of iterations.
