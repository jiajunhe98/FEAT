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

We also release the datasets used in our paper on 🤗 [Hugging Face](https://huggingface.co/datasets/JJHE/FEAT/). All samples in these datasets have been pre-aligned to reference configurations to facilitate mini-batch optimal transport (OT) pairing during training.

For custom datasets, especially for larger systems (ALDP, LJ-128, or larger), you may need to align your configurations first. See the alignment scripts in the `data/` folder (`align_rotation.py` and `align_rotation_permute.py`) and the [data folder README](data/readme.md) for more details.

---

### 🏃 Run FEAT

```
python main_train.py --config gmm_si > log.txt
```

hyparameters can be set in ```config/defaults/your-config.yaml```.

---

### 🚧 Coming soon:
- Code for Half-side interpolant

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
