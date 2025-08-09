from tqdm import tqdm
import torch
from loss import b_loss, v_loss, eta_loss, tsm_loss, ti_loss
from sampling import Jarzynski_integrate, Jarzynski_integrate_ODE
from utils import Coef, get_marginal_plot_fn
import matplotlib.pyplot as plt
import numpy as np
import time 

def update_ema(model, model_ema, rate):
    for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        param_ema.data = (1 - rate) * param.data + rate * param_ema.data


def train(vector_field, score_net, vector_field_ema, score_net_ema, ema_rate, optimizer, sampler: callable, target1, target2, batch_size, alpha, beta, gamma, learn_b, num_epochs=100, save_gap=10, save_dir=None, cfg=None, finetune=False, use_pyg=False, eval_div=True):
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    vector_field.train()
    score_net.train()
    losses = []

    # define n_particles and n_dim to center noise
    if cfg.target.name == 'gmm':
        center_noise = False
        n_particles = None
        n_dim = None
    elif cfg.target.name == 'aldp':
        center_noise = True
        n_particles = cfg.target.aldp1.n_particles
        n_dim = cfg.target.aldp1.n_dim
    elif cfg.target.name == 'lj':
        center_noise = True
        n_particles = cfg.target.lj1.n_particles
        n_dim = cfg.target.lj1.n_dim
    elif cfg.target.name == 'phi4':
        center_noise = False
        n_particles = None
        n_dim = None

    try:
        total_grad_accumulation = cfg.train.grad_accumulation
        print('Grad Accumulation:', total_grad_accumulation)
    except:
        total_grad_accumulation = 1
    grad_accumulation = 0

    for epoch in pbar:
        vector_field.train()
        score_net.train()

        


        if cfg.method != 'FM':
            
            # Sample data
            x1, x2, s1, s2 = sampler(batch_size)
            try:
                if cfg.train.data_aug:
                    # data augmentation
                    # generate random rotation matrix
                    from scipy.stats import special_ortho_group
                    x = torch.from_numpy(special_ortho_group.rvs(3)).float().to(x1.device)

                    x1 = (x1.reshape(x1.shape[0], -1, 3) @ x).reshape(x1.shape[0], -1)
                    x2 = (x2.reshape(x2.shape[0], -1, 3) @ x).reshape(x2.shape[0], -1)
                    s1 = (s1.reshape(s1.shape[0], -1, 3) @ x).reshape(s1.shape[0], -1)
                    s2 = (s2.reshape(s2.shape[0], -1, 3) @ x).reshape(s2.shape[0], -1)
            except:
                pass

            loss_1 = b_loss(x1, x2, alpha, beta, gamma, vector_field, center_noise, n_particles, n_dim) if learn_b else v_loss(x1, x2, alpha, beta, gamma, vector_field, center_noise, n_particles, n_dim)
            loss_2 = eta_loss(x1, x2, alpha, beta, gamma, score_net, center_noise, n_particles, n_dim)
            loss_3 = tsm_loss(x1, x2, s1, s2, alpha, beta, gamma, score_net, center_noise, n_particles, n_dim)
            loss = loss_1 + loss_2 + loss_3*0.1 # hard code for now
            try:
                
                if cfg.ti == True:
                    with torch.enable_grad():
                        loss += ti_loss(x1, x2, alpha, beta, gamma, score_net, center_noise, n_particles, n_dim) * 1e-8
                # rne_l = rne_loss(x1, x2, alpha, beta, gamma, vector_field, score_net, center_noise, n_particles, n_dim)
                # loss += rne_l * 1e-4
                print('USE TI REGULARIZATION')
            except:
                pass
        else:
            # Sample data
            x1, x2 = sampler(batch_size)
            loss_1 = b_loss(x1, x2, alpha, beta, gamma, vector_field, center_noise, n_particles, n_dim) if learn_b else v_loss(x1, x2, alpha, beta, gamma, vector_field, center_noise, n_particles, n_dim)
            loss = loss_1
        
        
        print (loss.item())
        # record loss, update and save ckpt
        losses.append(loss.item())
        if not torch.isnan(loss).any() and not torch.isinf(loss).any():
            
            loss.backward()
            grad_accumulation += 1

            if grad_accumulation % total_grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(vector_field.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(score_net.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
                grad_accumulation = 0

        else:
            print('Oh no! NAN!')
        update_ema(vector_field, vector_field_ema, ema_rate)
        update_ema(score_net, score_net_ema, ema_rate)
        if (epoch + 1) % 10 == 0:
            pbar.set_postfix({'Loss': loss.item()})
        if (epoch + 1) % save_gap == 0 and save_dir is not None:
            checkpoint = {
                'vector_field_state_dict': vector_field.state_dict(),
                'score_net_state_dict': score_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vector_field_ema_state_dict': vector_field_ema.state_dict(),
                'score_net_ema_state_dict': score_net_ema.state_dict(),
                'epoch': epoch,
                'losses': losses
            }
            torch.save(checkpoint, f"{save_dir}/checkpoint_epoch_{epoch + 1}.pth")


            # evaluate
            vector_field_ema.eval() 
            score_net_ema.eval()

            plt.plot(losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(f"{save_dir}/training_loss.png")
            plt.close()

            with torch.no_grad():
                
                if cfg.method != 'FM':
                    
                    x1, x2, s1, s2 = sampler(cfg.save.eval_sample, False)
                    print(cfg.save.eval_sample, x1.shape, x2.shape, s1.shape, s2.shape)

                    try:
                        if cfg.train.data_aug:
                            # data augmentation
                            # generate random rotation matrix
                            from scipy.stats import special_ortho_group
                            x = torch.from_numpy(special_ortho_group.rvs(3)).float().to(x1.device)

                            x1 = (x1.reshape(x1.shape[0], -1, 3) @ x).reshape(x1.shape[0], -1)
                            x2 = (x2.reshape(x2.shape[0], -1, 3) @ x).reshape(x2.shape[0], -1)
                            s1 = (s1.reshape(s1.shape[0], -1, 3) @ x).reshape(s1.shape[0], -1)
                            s2 = (s2.reshape(s2.shape[0], -1, 3) @ x).reshape(s2.shape[0], -1)
                    except:
                        pass

                else:
                    x1, x2 = sampler(cfg.save.eval_sample, False)


                    try:
                        if cfg.train.data_aug:
                            # data augmentation
                            # generate random rotation matrix
                            from scipy.stats import special_ortho_group
                            x = torch.from_numpy(special_ortho_group.rvs(3)).float().to(x1.device)

                            x1 = (x1.reshape(x1.shape[0], -1, 3) @ x).reshape(x1.shape[0], -1)
                            x2 = (x2.reshape(x2.shape[0], -1, 3) @ x).reshape(x2.shape[0], -1)
                    except:
                        pass


                if not use_pyg:
                    eval_batch_size = cfg.save.eval_batch
                    eval_batch_num = int(np.ceil(cfg.save.eval_sample / eval_batch_size))
                    logp1 = []
                    logp2 = []
                    for eb in tqdm(range(eval_batch_num)):
                        _x1 = x1[eb * eval_batch_size: (eb + 1) * eval_batch_size]
                        _x2 = x2[eb * eval_batch_size: (eb + 1) * eval_batch_size]
                        _logp1 = target1.log_prob(_x1)
                        _logp2 = target2.log_prob(_x2)
                        logp1.append(_logp1)
                        logp2.append(_logp2)
                    logp1 = torch.cat(logp1, 0)
                    logp2 = torch.cat(logp2, 0)

                else:
                    x1 = x1.to(cfg.device)
                    x2 = x2.to(cfg.device)
                    logp1, logp2 = target1.log_prob(x1.x.reshape(x1.batch_size, -1)), target2.log_prob(x2.x.reshape(x1.batch_size, -1)) 
                eps = Coef(cfg.coef.eps)
                times = torch.linspace(0, 1, cfg.sample.num_step).to(cfg.device)


                if cfg.method == 'FM':
                    time_now = time.time()
                    eval_batch_size = cfg.save.eval_batch
                    eval_batch_num = int(np.ceil(cfg.save.eval_sample / eval_batch_size))
                    x1p = []
                    A1 = []
                    x2p = []
                    A2 = []

                    for eb in tqdm(range(eval_batch_num)):
                        _x1p, _, _A1 = Jarzynski_integrate_ODE(x1.clone()[eb * eval_batch_size: (eb + 1) * eval_batch_size], 
                                                            vector_field=vector_field, 
                                                            n_steps=cfg.sample.num_step,
                                                            use_pyg=use_pyg,
                                                            calculate_div=eval_div)
                        _x2p, _, _A2 = Jarzynski_integrate_ODE(x2.clone()[eb * eval_batch_size: (eb + 1) * eval_batch_size], 
                                                            vector_field=vector_field, 
                                                            n_steps=cfg.sample.num_step,
                                                            forward=False,
                                                            use_pyg=use_pyg,
                                                            calculate_div=eval_div)
                        x1p.append(_x1p)
                        A1.append(_A1)
                        x2p.append(_x2p)
                        A2.append(_A2)
                    x1p = torch.cat(x1p, 0)
                    A1 = torch.cat(A1, 0)
                    x2p = torch.cat(x2p, 0)
                    A2 = torch.cat(A2, 0)
                    if eval_div:
                        if use_pyg:
                            DF1 = torch.logsumexp(A1 + target2.log_prob(x1p.x.reshape(x1.batch_size, -1)) - logp1, 0).item() - np.log(A1.shape[0])
                            DF2 = -torch.logsumexp(-A2 + target1.log_prob(x2p.x.reshape(x1.batch_size, -1)) - logp2, 0).item() + np.log(A2.shape[0])

                            DF_lower = torch.mean(A1 + target2.log_prob(x1p.x.reshape(x1.batch_size, -1)) - logp1, 0).item()
                            DF_upper = -torch.mean(-A2 - target1.log_prob(x2p.x.reshape(x1.batch_size, -1)) + logp2, 0).item()
                        else:
                            A1 = A1 + target2.log_prob(x1p) - logp1
                            A2 = A2 + target1.log_prob(x2p) - logp2

                            DF1 = torch.logsumexp(A1, 0).item() - np.log(A1.shape[0])
                            DF2 = -torch.logsumexp(A2, 0).item() + np.log(A2.shape[0])

                            DF_lower = torch.mean(A1, 0).item()
                            DF_upper = -torch.mean(A2, 0).item()
                            mv1 = (DF1 + DF2) / 2
                            for _ in range(10000):
                                mv1_new = (torch.logsumexp(torch.nn.LogSigmoid()(A1 - mv1), 0) - torch.logsumexp(torch.nn.LogSigmoid()(A2 + mv1), 0) + mv1).item()
                                if np.abs(mv1_new - mv1) < 1e-4:
                                    mv1 = mv1_new
                                    break
                                mv1 = mv1_new
                        with open(f"{save_dir}/F.txt", "a") as f:
                            f.writelines(f"Iter: {epoch + 1}, {DF1:.5f}, {DF2:.5f}, {DF_lower: .5f}, {DF_upper: .5f}, Min Var: {mv1: .5f}\n")
                    time_end = time.time()
                    with open(f"{save_dir}/F.txt", "a") as f:
                        f.writelines('Time:' + str(time_end - time_now) + '\n')

                    # plot marginals
                    evaluate = get_marginal_plot_fn(cfg)
                    if cfg.target.name in ['gmm', 'phi4']:
                        evaluate(x1.detach().cpu().numpy(), x2.detach().cpu().numpy(), 
                                x2p.detach().cpu().numpy(), x1p.detach().cpu().numpy(), 
                                f"{save_dir}/sample_{epoch + 1}.png")
                    elif cfg.target.name in ['lj', 'aldp',]:
                        if use_pyg:
                            x1 = x1.x.reshape(x1.batch_size, -1).detach()
                            x2 = x2.x.reshape(x1.batch_size, -1).detach()
                            x1p = x1p.x.reshape(x1.batch_size, -1).detach()
                            x2p = x2p.x.reshape(x1.batch_size, -1).detach()
                        else:
                            x1 = x1.detach()
                            x2 = x2.detach()
                            x1p = x1p.detach()
                            x2p = x2p.detach()

                        evaluate(x1, 
                                    x2,
                                    x2p,
                                    x1p,
                                    f"{save_dir}/sample1_{epoch + 1}.png",
                                    target1, 
                                    target2,)
                    
                else:
                    time_now = time.time()
                    eval_batch_size = cfg.save.eval_batch
                    eval_batch_num = int(np.ceil(cfg.save.eval_sample / eval_batch_size))
                    x1p = []
                    A1 = []
                    x2p = []
                    A2 = []
                    print(eval_batch_size, eval_batch_num)
                    for eb in tqdm(range(eval_batch_num)):
                        _x1p, _, _A1 = Jarzynski_integrate(
                                                        x_init=x1.clone()[eb * eval_batch_size: (eb + 1) * eval_batch_size],
                                                        init_logprob=logp1.clone()[eb * eval_batch_size: (eb + 1) * eval_batch_size], 
                                                        eps=eps, 
                                                        alpha=alpha,
                                                        beta=beta,
                                                        gamma=gamma, 
                                                        vector_field=vector_field_ema, 
                                                        score_net=score_net_ema, 
                                                        collect_interval=100, 
                                                        learn_b=learn_b, 
                                                        times=times,
                                                        forward=True,
                                                        return_A=True,
                                                        target_logp=target2.log_prob,
                                                        center_noise=center_noise,
                                                        n_particles=n_particles,
                                                        n_dim=n_dim
                                                        )
                        _x2p, _, _A2 = Jarzynski_integrate(x_init=x2.clone()[eb * eval_batch_size: (eb + 1) * eval_batch_size],
                                                        init_logprob=logp2.clone()[eb * eval_batch_size: (eb + 1) * eval_batch_size], 
                                                        eps=eps, 
                                                        alpha=alpha,
                                                        beta=beta,
                                                        gamma=gamma, 
                                                        vector_field=vector_field_ema, 
                                                        score_net=score_net_ema, 
                                                        collect_interval=100, 
                                                        learn_b=learn_b, 
                                                        times=times.flip(0),
                                                        forward=False,
                                                        return_A=True,
                                                        target_logp=target1.log_prob,
                                                        center_noise=center_noise,
                                                        n_particles=n_particles,
                                                        n_dim=n_dim)
                        x1p.append(_x1p)
                        A1.append(_A1)
                        x2p.append(_x2p)
                        A2.append(_A2)
                    x1p = torch.cat(x1p, 0)
                    A1 = torch.cat(A1, 0)
                    x2p = torch.cat(x2p, 0)
                    A2 = torch.cat(A2, 0)

                    A1 = A1[A1.isnan() == False]
                    A2 = A2[A2.isnan() == False]

                    

                    DF1 = torch.logsumexp(A1, 0).item() - np.log(A1.shape[0])
                    DF2 = -torch.logsumexp(A2, 0).item() + np.log(A2.shape[0])     
                    DF_lower = A1.mean().item()
                    DF_upper = -A2.mean().item()

                    mv1 = (DF1 + DF2) / 2
                    for iii in range(10000):
                        mv1_new = (torch.logsumexp(torch.nn.LogSigmoid()(A1 - mv1), 0) - torch.logsumexp(torch.nn.LogSigmoid()(A2 + mv1), 0) + mv1).item()
                        if np.abs(mv1_new - mv1) < 1e-4:
                            mv1 = mv1_new
                            break
                        mv1 = mv1_new
                        # print(mv1)
                    with open(f"{save_dir}/F.txt", "a") as f:
                        f.writelines(f"Iter: {epoch + 1}, {DF1:.5f}, {DF2:.5f}, {DF_lower: .5f}, {DF_upper: .5f}, Min Var: {mv1: .5f}\n")
                    time_end = time.time()
                    with open(f"{save_dir}/F.txt", "a") as f:
                        f.writelines('Time:' + str(time_end - time_now) + '\n')

                    # plot marginals
                    evaluate = get_marginal_plot_fn(cfg)
                    if cfg.target.name in ['gmm', 'phi4']:
                        evaluate(x1.detach().cpu().numpy(), x2.detach().cpu().numpy(), 
                                x2p.detach().cpu().numpy(), x1p.detach().cpu().numpy(), 
                                f"{save_dir}/sample_{epoch + 1}.png")
                    elif cfg.target.name in ['lj', 'aldp']:
                        if use_pyg:
                            x1 = x1.x.reshape(x1.batch_size, -1).detach()
                            x2 = x2.x.reshape(x1.batch_size, -1).detach()
                            x1p = x1p.x.reshape(x1.batch_size, -1).detach()
                            x2p = x2p.x.reshape(x1.batch_size, -1).detach()
                        else:
                            x1 = x1.detach()
                            x2 = x2.detach()
                            x1p = x1p.detach()
                            x2p = x2p.detach()

                        evaluate(x1, 
                                    x2,
                                    x2p,
                                    x1p,
                                    f"{save_dir}/sample1_{epoch + 1}.png",
                                    target1, 
                                    target2,)
    return losses
