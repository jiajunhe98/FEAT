# FEAT Code


Please put the data in ```data/``` folder. As an example, you can download the dataset we use for ALDP-S and LJ55/79 at [this anonymous drive link](https://drive.google.com/drive/folders/1Ujsjp7qNR3qwPcqWVrX5lORB7CM8fK_o?usp=sharing).


Then, run

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

"# FEAT" 
