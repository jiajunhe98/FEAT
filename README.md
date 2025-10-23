<div align="left">
**NeurIPS 2025**
</div>
<h1 align="center">FEAT: Free energy Estimators with Adaptive Transport</h1>
<div align="left">
Jiajun He*, Yuanqi Du*, Francisco Vargas, Carla P. Gomes, José Miguel Hernández-Lobato, Eric Vanden-Eijnden  
  
*Equal Contribution
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2504.11516-b31b1b.svg)](https://arxiv.org/abs/2504.11516)



🚧 Under construction and coming soon:
- instruction for environment installation
- Code for Half-side interpolant and FEAT for DeepBAR
- Data release

We aim to finish construction soon. If you have any questions regarding the experiments, please feel free to reach out!



### Data preparation

Please put the data in ```data/``` folder.


### Run FEAT

```
python main_train.py --config gmm_si > log.txt
```

hyparameters can be set in ```config/defaults/xxx.yaml```.



### Env

Our code is implemented in ```pytorch```. After preparing the environment with torch, please also install openmm and other dependencies

```
conda install -c conda-forge openmm openmmtools
pip install normflows
pip install git+https://github.com/VincentStimper/boltzmann-generators.git
```
and install bgflow at [https://github.com/noegroup/bgflow](https://github.com/noegroup/bgflow).

For experiments with OT plan, please install ```torchcfm``` package via

```
pip install torchcfm
```

### Reference
```
@article{he2025feat,
  title={FEAT: Free energy Estimators with Adaptive Transport},
  author={He*, Jiajun and Du*, Yuanqi and Vargas, Francisco and Wang, Yuanqing and Gomes, Carla P and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel and Vanden-Eijnden, Eric},
  journal={arXiv preprint arXiv:2504.11516},
  year={2025}
}
```
