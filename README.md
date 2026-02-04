<h1 style="margin-bottom:0;">FEAT: Free energy Estimators with Adaptive Transport</h1>
<p style="margin-top:2px;">
  Jiajun He*, Yuanqi Du*, Francisco Vargas, Carla P. Gomes, José Miguel Hernández-Lobato, Eric Vanden-Eijnden<br>
  <em>*Equal Contribution</em>
</p>

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-00b3b3.svg)]([https://neurips.cc/](https://neurips.cc/virtual/2025/poster/118966)) [![arXiv](https://img.shields.io/badge/arXiv-2504.11516-b31b1b.svg)](https://arxiv.org/abs/2504.11516) 


---




### ⚙️ Environment

Our implementation is based on [PyTorch](https://pytorch.org/). Torch ```2.9.1+cu128``` works well; other versions may also be compatible..

After setting up PyTorch, please install the following dependencies:

```
# Core molecular simulation libraries
conda install -c conda-forge openmm openmmtools

# Normalizing flow and Boltzmann generator components
pip install normflows
pip install git+https://github.com/VincentStimper/boltzmann-generators.git

# Conditional flow matching library
pip install torchcfm

```
Finally, install bgflow manually from the official repository: [https://github.com/noegroup/bgflow](https://github.com/noegroup/bgflow).


---


### 📁 Data preparation

Place your data files in the `data/` folder. 

We also release the datasets used in our paper on 🤗 [Hugging Face](https://huggingface.co/datasets/JJHE/FEAT/). Samples for ALDP, LJ-79/128 in these datasets have been pre-aligned to reference configurations to facilitate mini-batch optimal transport (OT) pairing during training.

For custom datasets, especially for larger systems (ALDP, LJ-128, or larger), you may need to align your configurations first. See the alignment scripts in the `data/` folder (`align_rotation.py` and `align_rotation_permute.py`) and the [data folder README](data/readme.md) for more details.

---

### 🏃 Run Standard FEAT

To train FEAT, run:

```bash
python main_train.py --config gmm_si > log.txt
```

Hyperparameters can be configured in the YAML files in the `config/` directory. Available configuration files include:
- `gmm_si.yaml` / `gmm_fm.yaml` - Gaussian mixture model (SI: Schrödinger interpolant, FM: Flow matching)
- `aldp_si.yaml` / `aldp_fm.yaml` - Alanine dipeptide
- `lj79_si.yaml` / `lj79_fm.yaml` - Lennard-Jones 79-particle system
- `phi_si.yaml` - φ⁴ field theory

Default hyperparameter templates are available in `config/defaults/`.

---

### 🏃 Run One-sided FEAT

We provide scripts for training and inference of One-sided FEAT with EDM parametried diffusion models.
In our paper, we use one-sided FEAT for alanine tetrapeptide and Chignolin.

#### Training

To train the model:

```bash
# Basic training
python train_dm.py --data_path data/data.pt --n_particles 43

# Full training with custom parameters
python train_dm.py \
    --data_path data/data.pt \
    --n_particles 43 \
    --data_scaling 5.0 \
    --n_epochs 100000 \
    --batch_size 20 \
    --lr 1e-4 \
    --save_dir ./checkpoints
```

#### Inference with Jarzynski Equality

To estimate free energy differences using the trained model:

```bash
# Basic inference (reverse path only)
python inference_dm.py --data_path data/data.pt --model_path checkpoints/ema_net_final.pt --n_particles 43

# Full inference with forward and reverse paths
python inference_dm.py \
    --model_path checkpoints/ema_net_final.pt \
    --n_particles 43 \
    --data_path data/data.pt \
    --data_scaling 5.0 \
    --n_samples 100 \
    --n_batches 10
```

For more details, see the docstrings in `train_dm.py` and `inference_dm.py`.

---

### 📧 Support and Contact
If you have any questions, please feel free to reach out at jh2383@cam.ac.uk

---

### 📍 Reference
```
@inproceedings{he2025feat,
  title     = {FEAT: Free energy Estimators with Adaptive Transport},
  author    = {He, Jiajun and Du, Yuanqi and Vargas, Francisco and Wang, Yuanqing and Gomes, Carla P. and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel and Vanden-Eijnden, Eric},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
}

```
