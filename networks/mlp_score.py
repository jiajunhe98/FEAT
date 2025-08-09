from torch import nn
import torch
import numpy as np


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        x = (x + 1e-1).log() / 4
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    

class score_PotentialNet(nn.Module):
    
    def __init__(self, 
                 dim, 
                 device,
                 alpha, beta, gamma, 
                 num_layers: int = 4,
                 num_hid: int = 256,
                 clip: float =1e4,
                 *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_hid = num_hid
        self.clip = clip

        self.time_embed = PositionalEmbedding(self.num_hid)

        self.time_coder_state = nn.Sequential(*[
            nn.Linear(self.num_hid, self.num_hid),
            nn.GELU(),
            nn.Linear(self.num_hid, self.num_hid),
        ])
        self.state_time_net = [nn.Sequential(
            *[nn.Linear(self.num_hid+self.dim, self.num_hid), nn.GELU()])] + [nn.Sequential(
            *[nn.Linear(self.num_hid, self.num_hid), nn.GELU()]) for _ in range(self.num_layers-1)] + [
                                                nn.Linear(self.num_hid, self.dim)]
        self.state_time_net = nn.Sequential(*self.state_time_net)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _forward(self, input_array, time_array, *args, **kwargs):
        time_array_emb = self.time_embed(time_array)
        t_net1 = self.time_coder_state(time_array_emb)

        extended_input = torch.cat((input_array, t_net1), -1)
        out_state = self.state_time_net(extended_input)

        return out_state

    def score(self, input_array, time_array, clip=False, *args, **kwargs):
        return self._forward(input_array, time_array)
    
    def forward(self, input_array, time_array, *args, **kwargs):
        raise NotImplementedError("MW_PotentialNet does not support forward method, use score method instead.")


class score_VectorFieldNet(nn.Module):
    
    def __init__(self, 
                 dim, 
                 device,
                 num_layers: int = 4,
                 num_hid: int = 256,
                 clip: float = 1e4,
                 *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_hid = num_hid
        self.clip = clip


        self.time_embed = PositionalEmbedding(self.num_hid)

        self.time_coder_state = nn.Sequential(*[
            nn.Linear(self.num_hid, self.num_hid),
            nn.GELU(),
            nn.Linear(self.num_hid, self.num_hid),
        ])
        self.state_time_net = [nn.Sequential(
            *[nn.Linear(self.num_hid+self.dim, self.num_hid), nn.GELU()])] + [nn.Sequential(
            *[nn.Linear(self.num_hid, self.num_hid), nn.GELU()]) for _ in range(self.num_layers-1)] + [
                                                nn.Linear(self.num_hid, self.dim)]
        self.state_time_net = nn.Sequential(*self.state_time_net)


    def forward(self, input_array, time_array, *args, **kwargs):
        time_array_emb = self.time_embed(time_array)
        t_net1 = self.time_coder_state(time_array_emb)

        extended_input = torch.cat((input_array, t_net1), -1)
        out_state = self.state_time_net(extended_input)
        return torch.clip(out_state, -self.clip, self.clip)
    
    def div(self, input_array, time_array, *args, **kwargs):
        assert len(input_array.shape) == 2
        #  input_array: (batch_size, dim)
        with torch.enable_grad():
            input_array = input_array.requires_grad_(True)
            out = torch.autograd.functional.jacobian(lambda x: self.forward(x, time_array).sum(0), input_array, create_graph=True, vectorize=True)
            div = out.diagonal(offset=0, dim1=-1, dim2=-3).sum(-1)
            return div
    

class MLP(nn.Module):
    
    def __init__(self, 
                 dim, 
                 num_layers: int = 2,
                 num_hid: int = 64,
                 *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_hid = num_hid

        self.net = [nn.Sequential(
            *[nn.Linear(self.dim, self.num_hid), nn.GELU()])] + [nn.Sequential(
            *[nn.Linear(self.num_hid, self.num_hid), nn.GELU()]) for _ in range(self.num_layers-1)] + [
                                                nn.Linear(self.num_hid, self.dim)]
        self.net = nn.Sequential(*self.net)

    def forward(self, input_array, *args, **kwargs):
        out_state = self.net(input_array)
        return out_state
    