import torch
from networks.mlp import PotentialNet, VectorFieldNet
from networks.mlp_score import score_PotentialNet, score_VectorFieldNet
from networks.egnn import EGNNPotentialNet, EGNNVectorFieldNet, EGNNPotentialNet_ALDP, EGNNVectorFieldNet_ALDP
from utils import Coef
import argparse
import yaml
from types import SimpleNamespace
from utils import get_target, get_sampler_from_dataset, get_sampler_from_samples, get_sampler_from_target, get_sampler_with_grad_from_target, get_sampler_with_grad_fix_order
from training import train
import os
import matplotlib.pyplot as plt
from copy import deepcopy

def get_unique_dir(base_dir):
    if not os.path.exists(base_dir):
        return base_dir
    counter = 1
    new_dir = f"{base_dir}_{counter}"
    while os.path.exists(new_dir):
        counter += 1
        new_dir = f"{base_dir}_{counter}"
    return new_dir

def main(cfg):
    
    target = cfg.target.name

    # make dir
    save_dir = cfg.save.save_dir.strip('/')
    os.makedirs(save_dir, exist_ok=True)

    save_dir = save_dir + '/' + target
    save_dir = get_unique_dir(save_dir)
    os.makedirs(save_dir, exist_ok=True)


    def namespace_to_dict(ns):
        if isinstance(ns, SimpleNamespace):
            return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
        elif isinstance(ns, list):
            return [namespace_to_dict(v) for v in ns]
        else:
            return ns
    cfg_dict = namespace_to_dict(cfg)
    # Save to a YAML file
    with open(save_dir + '/' + "config.yaml", "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
        

    # Set up
    dim = cfg.target.dim
    device = cfg.device
    learn_b = cfg.network.learn_b #  learn b or v

    # Set coefficients
    alpha = Coef(cfg.coef.alpha)
    beta = Coef(cfg.coef.beta)
    gamma = Coef(cfg.coef.gamma, cfg.coef.a, cfg.coef.b)
    if cfg.method == 'FM':
        gamma = Coef(cfg.coef.gamma, 0) # if FM, there will be no gamma term
        assert learn_b == True # if FM, learn b and v will make no difference, but in our implementation, we first map v back to b using the score net, so we need to learn b

    # Define targets at both ends
    target1, target2 = get_target(cfg)
    sampler = get_sampler_from_target(cfg, target1, target2)
    
    if cfg.method == 'SI':
        sampler = get_sampler_with_grad_from_target(cfg, target1, target2)


    if cfg.network.name == 'egnn':
    
            
        if cfg.target.name == 'aldp':
            score_net = EGNNPotentialNet_ALDP(cfg.target.aldp1.n_particles,
                                              cfg.target.aldp1.n_dim,
                                              hidden_nf=cfg.network.score_net.num_hid,
                                              act_fn=torch.nn.SiLU(),
                                              n_layers=cfg.network.score_net.num_layer,
                                            #   cutoff=cfg.network.cutoff,
                                              recurrent=True,
                                              attention=True,
                                              tanh=True,
                                              device=device).to(device)
            vector_field = EGNNVectorFieldNet_ALDP(cfg.target.aldp1.n_particles,
                                                   cfg.target.aldp1.n_dim,
                                                   hidden_nf=cfg.network.score_net.num_hid,
                                                   act_fn=torch.nn.SiLU(),
                                                   n_layers=cfg.network.score_net.num_layer,
                                                #    cutoff=cfg.network.cutoff,
                                                   recurrent=True,
                                                   attention=True,
                                                   tanh=True,
                                                   device=device).to(device)

            
        elif cfg.target.name == 'lj':
            score_net = EGNNPotentialNet(cfg.target.lj1.n_particles,
                                        cfg.target.lj1.n_dim,
                                        hidden_nf=cfg.network.score_net.num_hid,
                                        act_fn=torch.nn.ReLU(),
                                        n_layers=cfg.network.score_net.num_layer,
                                        cutoff=cfg.network.cutoff,
                                        recurrent=True,
                                        attention=True,
                                        tanh=True,).to(device)
            vector_field = EGNNVectorFieldNet(cfg.target.lj1.n_particles,
                                            cfg.target.lj1.n_dim,
                                            hidden_nf=cfg.network.score_net.num_hid,
                                            act_fn=torch.nn.ReLU(),
                                            n_layers=cfg.network.score_net.num_layer,
                                            cutoff=cfg.network.cutoff,
                                            recurrent=True,
                                            attention=True,
                                            tanh=True,).to(device)

    
    else:
        score_net = score_PotentialNet(dim, device, alpha, beta, gamma, cfg.network.score_net.num_layer, cfg.network.score_net.num_hid, cfg.network.score_net.clip).to(device)
        vector_field = score_VectorFieldNet(dim, device, cfg.network.vector_field.num_layer, cfg.network.vector_field.num_hid, cfg.network.vector_field.clip).to(device)

    if cfg.method == 'FM':
        optimizer = torch.optim.Adam(vector_field.parameters(), lr=cfg.train.lr)
    else:
        optimizer = torch.optim.Adam(list(score_net.parameters()) + list(vector_field.parameters()) , lr=cfg.train.lr)
    
    if cfg.train.finetune:
        print('Load ckpt from ', cfg.train.pretrain_ckpt)
        ckpt = torch.load(cfg.train.pretrain_ckpt, map_location=device)
        score_net.load_state_dict(ckpt['score_net_ema_state_dict'], strict=False)
        vector_field.load_state_dict(ckpt['vector_field_ema_state_dict'], strict=False)
        if not cfg.train.finetune_vector_field:
            vector_field.requires_grad_(False)
        if not cfg.train.finetune_score_net:
            score_net.requires_grad_(False)

    score_net_ema = deepcopy(score_net).requires_grad_(False)
    vector_field_ema = deepcopy(vector_field).requires_grad_(False)
    try:
        eval_div = cfg.save.eval_div
    except:
        eval_div = False
    losses = train(vector_field, 
                   score_net, 
                   vector_field_ema,
                   score_net_ema,
                   cfg.train.ema_rate,
                   optimizer, 
                   sampler=sampler,
                   target1=target1,
                   target2=target2,
                   batch_size=cfg.train.batch_size,
                   alpha=alpha, 
                   beta=beta, 
                   gamma=gamma, 
                   learn_b=learn_b, 
                   num_epochs=cfg.train.num_epoch, 
                   save_gap=cfg.save.eval_gap, 
                   save_dir=save_dir, 
                   cfg=cfg,
                   finetune=cfg.train.finetune,
                   use_pyg=(cfg.network.name == 'gnn'),
                   eval_div=eval_div,)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    plt.close()

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load configuration file")
    parser.add_argument('--config', type=str, help='Name of the configuration file')
    args = parser.parse_args()

    with open('config/' + args.config + '.yaml', 'r') as file:
        cfg_override = yaml.safe_load(file)
    with open('config/' + cfg_override['default'], 'r') as file:
        cfg_default = yaml.safe_load(file)    

    def recursive_update(default, override):
        """ Recursively update default dictionary with override dictionary """
        for key, value in override.items():
            if isinstance(value, dict) and key in default and isinstance(default[key], dict):
                default[key] = recursive_update(default[key], value)
            else:
                default[key] = value
        return default

    cfg_merged = recursive_update(cfg_default, cfg_override)
    
    cfg = dict_to_namespace(cfg_merged)
    


    main(cfg)