import torch
import torch.nn as nn
from bgflow.utils import distance_vectors, distances_from_vectors

class EGNN_dynamics(nn.Module):
    def __init__(
        self,
        n_particles,
        n_dimension,
        hidden_nf=64,
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        tanh=False,
        agg="sum",
        remove_input=True,
        cutoff=None,
    ):
        super().__init__()
        self.egnn = EGNN(
            in_node_nf=1,
            in_edge_nf=1,
            hidden_nf=hidden_nf,
            act_fn=act_fn,
            n_layers=n_layers,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            agg=agg,
        )

        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.cutoff = cutoff
        if self.cutoff is None:
            self.edges = self._create_edges()
        self._edges_dict = {}

        # Count function calls
        self.counter = 0
        self.remove_input = remove_input

    def _forward(self, input_array, time_array, *args, **kwargs):
        

        t = time_array
        xs = input_array

        n_batch = xs.shape[0]

        if self.cutoff is not None:
            edges = self._create_edges(xs)
        else:
            edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = [edges[0].to(xs.device), edges[1].to(xs.device)]
        x = xs.reshape(n_batch * self._n_particles, self._n_dimension).clone()
        h = torch.ones(n_batch, self._n_particles).to(xs.device)

        t = t.unsqueeze(-1)
        h = h * t

        h = h.reshape(n_batch * self._n_particles, 1)
        edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)

        vel = x_final - x if self.remove_input else x_final

        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        vel = remove_mean(vel, self._n_particles, self._n_dimension)
        self.counter += 1
        return vel.view(n_batch, self._n_particles * self._n_dimension)

    def _create_edges(self, xs=None):
        if xs is None:
            rows, cols = [], []
            for i in range(self._n_particles):
                for j in range(i + 1, self._n_particles):
                    rows.append(i)
                    cols.append(j)
                    rows.append(j)
                    cols.append(i)
            return [torch.LongTensor(rows), torch.LongTensor(cols)]
        else:
            n_batch = xs.shape[0] 
            xs = xs.view(n_batch, self._n_particles, self._n_dimension)
            dist_matrix = torch.sqrt(torch.sum((xs.unsqueeze(2) - xs.unsqueeze(1)) ** 2, dim=-1))
            edge_list = [[], []]
            if n_batch not in self._edges_dict:
                self._edges_dict = {}
            for b in range(n_batch):
                index = torch.nonzero(dist_matrix[b] < self.cutoff, as_tuple=False)
                i, j = index[:, 0].cpu(), index[:, 1].cpu()
                edge_list[0].append(i + b * self._n_particles)
                edge_list[0].append(j + b * self._n_particles)
                edge_list[1].append(j + b * self._n_particles)
                edge_list[1].append(i + b * self._n_particles)
            edge_list[0] = torch.cat(edge_list[0])
            edge_list[1] = torch.cat(edge_list[1])
            self._edges_dict[n_batch] = [edge_list[0], edge_list[1]]
            return self._edges_dict[n_batch]
        
    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total)
            cols_total = torch.cat(cols_total)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total)
            cols_total = torch.cat(cols_total)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]


class EGNN_dynamics_AD2(nn.Module):
    def __init__(self, n_particles, n_dimension, hidden_nf=64, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', agg='sum'):
        super().__init__()
        self.mode = mode
        # Initial one hot encoding of the different element types for ALDP
        # following TBG code
        atom_types = torch.arange(22)
        # atom_types[[0, 2, 3]] = 2
        # atom_types[[19, 20, 21]] = 20
        # atom_types[[11, 12, 13]] = 12
        h_initial = torch.nn.functional.one_hot(atom_types)
        self.h_initial = h_initial

        if mode == 'egnn_dynamics':
            h_size = h_initial.size(1)
            if condition_time:
                h_size += 1
             
            self.egnn = EGNN(in_node_nf=h_size, in_edge_nf=1, hidden_nf=hidden_nf, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
        else:
            raise NotImplemented()

        self.device = device
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        # Count function calls
        self.counter = 0
        
    def _forward(self, input_array, time_array, *args, **kwargs):
        t = time_array[:, None]
        xs = input_array

        n_batch = xs.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = [edges[0], edges[1]]
        #Changed by Leon
        x = xs.reshape(n_batch*self._n_particles, self._n_dimension).clone()
        h = self.h_initial.to(self.device).reshape(1,-1)
        h = h.repeat(n_batch, 1)
        h = h.reshape(n_batch*self._n_particles, -1)
        # node compatability
        t = torch.tensor(t).to(xs)
        if t.shape != (n_batch,1):
            t = t.repeat(n_batch)
        t = t.repeat(1, self._n_particles)
        t = t.reshape(n_batch*self._n_particles, 1)
        if self.condition_time:
            h = torch.cat([h, t], dim=-1)
        if self.mode == 'egnn_dynamics':
            edge_attr = torch.sum((x[edges[0]] - x[edges[1]])**2, dim=1, keepdim=True)
            _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
            vel = x_final - x

        else:
            raise NotImplemented()
            
        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        vel = remove_mean(vel, self._n_particles, self._n_dimension)
        #print(t, xs)
        self.counter += 1
        return vel.view(n_batch,  self._n_particles* self._n_dimension)

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(self.device)
            cols_total = torch.cat(cols_total).to(self.device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]
    



class EGNN_dynamics_AD4(nn.Module):
    def __init__(self, n_particles, n_dimension, hidden_nf=64, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', agg='sum'):
        super().__init__()
        self.mode = mode
        # Initial one hot encoding of the different element types for ALDP
        atom_types = [
            0,   # 0: N (unique)
            1,   # 1: H1 (sym group 1)
            1,   # 2: H2 (sym group 1)
            1,   # 3: H3 (sym group 1)
            2,   # 4: CA
            3,   # 5: HA
            4,   # 6: CB
            5,   # 7: HB1 (sym group 2)
            5,   # 8: HB2 (sym group 2)
            5,   # 9: HB3 (sym group 2)
            6,   # 10: C
            7,   # 11: O

            8,   # 12: N
            9,   # 13: H
            10,   # 14: CA
            11,   # 15: HA
            12,   # 16: CB
            13,   # 17: HB1 (sym group 3)
            13,   # 18: HB2 (sym group 3)
            13,   # 19: HB3 (sym group 3)
            14,   # 20: C
            15,   # 21: O

            16,   # 22: N
            17,   # 23: H
            18,   # 24: CA
            19,   # 25: HA
            20,   # 26: CB
            21,   # 27: HB1 (sym group 4)
            21,   # 28: HB2 (sym group 4)
            21,   # 29: HB3 (sym group 4)
            22,   # 30: C
            23,   # 31: O

            24,   # 32: N
            25,   # 33: H
            26,   # 34: CA
            27,   # 35: HA
            28,   # 36: CB
            29,   # 37: HB1 (sym group 5)
            29,   # 38: HB2 (sym group 5)
            29,   # 39: HB3 (sym group 5)
            30,   # 40: C
            31,   # 41: O
            32,   # 42: OXT
        ]
        atom_types = torch.tensor(atom_types, dtype=torch.long)


        # torch.arange(43)
        h_initial = torch.nn.functional.one_hot(atom_types)
        self.h_initial = h_initial

        if mode == 'egnn_dynamics':
            h_size = h_initial.size(1)
            if condition_time:
                h_size += 1
             
            self.egnn = EGNN(in_node_nf=h_size, in_edge_nf=1, hidden_nf=hidden_nf, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
        else:
            raise NotImplemented()

        self.device = device
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        # Count function calls
        self.counter = 0
        
    def _forward(self, input_array, time_array, *args, **kwargs):
        t = time_array[:, None]
        xs = input_array

        n_batch = xs.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = [edges[0], edges[1]]
        #Changed by Leon
        x = xs.reshape(n_batch*self._n_particles, self._n_dimension).clone()
        h = self.h_initial.to(self.device).reshape(1,-1)
        h = h.repeat(n_batch, 1)
        h = h.reshape(n_batch*self._n_particles, -1)
        # node compatability
        t = torch.tensor(t).to(xs)
        if t.shape != (n_batch,1):
            t = t.repeat(n_batch)
        t = t.repeat(1, self._n_particles)
        t = t.reshape(n_batch*self._n_particles, 1)
        if self.condition_time:
            h = torch.cat([h, t], dim=-1)
        if self.mode == 'egnn_dynamics':
            edge_attr = torch.sum((x[edges[0]] - x[edges[1]])**2, dim=1, keepdim=True)
            _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
            vel = x_final - x

        else:
            raise NotImplemented()
            
        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        vel = remove_mean(vel, self._n_particles, self._n_dimension)
        #print(t, xs)
        self.counter += 1
        return vel.view(n_batch,  self._n_particles* self._n_dimension)

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(self.device)
            cols_total = torch.cat(cols_total).to(self.device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]


class EGNNPotentialNet_AD4(EGNN_dynamics_AD4):
    def score(self, input_array, time_array, *args, **kwargs):
        input_array = remove_mean(input_array, self._n_particles, self._n_dimension)
        return remove_mean(self._forward(input_array, time_array, *args, **kwargs), self._n_particles, self._n_dimension)
    def forward(self, input_array, time_array, *args, **kwargs):
        raise NotImplementedError("This function should not be called for PotentialNet")
    

class EGNNPotentialNet_ALDP(EGNN_dynamics_AD2):
    def score(self, input_array, time_array, *args, **kwargs):
        input_array = remove_mean(input_array, self._n_particles, self._n_dimension)
        return remove_mean(self._forward(input_array, time_array, *args, **kwargs), self._n_particles, self._n_dimension)
    def forward(self, input_array, time_array, *args, **kwargs):
        raise NotImplementedError("This function should not be called for PotentialNet")
 

class EGNNVectorFieldNet_ALDP(EGNN_dynamics_AD2):
    def forward(self, input_array, time_array, *args, **kwargs):
        input_array = remove_mean(input_array, self._n_particles, self._n_dimension)
        return remove_mean(self._forward(input_array, time_array, *args, **kwargs), self._n_particles, self._n_dimension)
    
    def div(self, input_array, time_array, *args, **kwargs):
        input_array = remove_mean(input_array, self._n_particles, self._n_dimension)
        assert len(input_array.shape) == 2
        #  input_array: (batch_size, dim)
        with torch.enable_grad():
            input_array = input_array.requires_grad_(True)
            out = torch.autograd.functional.jacobian(lambda x: self.forward(x, time_array).sum(0), input_array, create_graph=True, vectorize=True)
            div = out.diagonal(offset=0, dim1=-1, dim2=-3).sum(-1)
            return div

class EGNNVectorFieldNet_AD4(EGNN_dynamics_AD4):
    def forward(self, input_array, time_array, *args, **kwargs):
        input_array = remove_mean(input_array, self._n_particles, self._n_dimension)
        return remove_mean(self._forward(input_array, time_array, *args, **kwargs), self._n_particles, self._n_dimension)
    
    def div(self, input_array, time_array, *args, **kwargs):
        input_array = remove_mean(input_array, self._n_particles, self._n_dimension)
        assert len(input_array.shape) == 2
        #  input_array: (batch_size, dim)
        with torch.enable_grad():
            input_array = input_array.requires_grad_(True)
            out = torch.autograd.functional.jacobian(lambda x: self.forward(x, time_array).sum(0), input_array, create_graph=True, vectorize=True)
            div = out.diagonal(offset=0, dim1=-1, dim2=-3).sum(-1)
            return div


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
    

class rbf_emb(nn.Module):
    def __init__(self, num_rbf, soft_cutoff_upper, rbf_trainable=False):
        super().__init__()
        self.soft_cutoff_upper = soft_cutoff_upper
        self.soft_cutoff_lower = 0
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()

        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.soft_cutoff_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.soft_cutoff_lower))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (end_value - start_value))**-2] *
                             self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        # dist=dist.squeeze(-1)
        soft_cutoff = 0.5 * \
                  (torch.cos(dist * torch.pi / self.soft_cutoff_upper) + 1.0)
        soft_cutoff = soft_cutoff * (dist < self.soft_cutoff_upper).float()
        return soft_cutoff*torch.exp(-self.betas * torch.square((torch.exp(-dist) - self.means)))
   
   


class EGNNPotentialNet(EGNN_dynamics):
    def score(self, input_array, time_array, *args, **kwargs):
        input_array = remove_mean(input_array, self._n_particles, self._n_dimension)
        return remove_mean(self._forward(input_array, time_array, *args, **kwargs), self._n_particles, self._n_dimension)
    def forward(self, input_array, time_array, *args, **kwargs):
        raise NotImplementedError("This function should not be called for PotentialNet")

    

class EGNNVectorFieldNet(EGNN_dynamics):
    def forward(self, input_array, time_array, *args, **kwargs):
        input_array = remove_mean(input_array, self._n_particles, self._n_dimension)
        return remove_mean(self._forward(input_array, time_array, *args, **kwargs), self._n_particles, self._n_dimension)
    
    def div(self, input_array, time_array, *args, **kwargs):
        input_array = remove_mean(input_array, self._n_particles, self._n_dimension)
        assert len(input_array.shape) == 2
        #  input_array: (batch_size, dim)
        with torch.enable_grad():
            input_array = input_array.requires_grad_(True)
            out = torch.autograd.functional.jacobian(lambda x: self.forward(x, time_array).sum(0), input_array, create_graph=True, vectorize=True)
            div = out.diagonal(offset=0, dim1=-1, dim2=-3).sum(-1)
            return div
    


class TimeEmbedding(nn.Module):
    def __init__(self, t_embed_dim, scale=30.0):
        super().__init__()
        self.register_buffer("w", torch.randn(t_embed_dim//2)*scale)

    def forward(self, t):
        # t: (B, )
        t_proj = 2.0 * torch.pi * self.w[None, :] * t[:, None]  # (B, E//2)
        t_embed = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)  # (B, E)
        return t_embed
    

class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        act_fn=nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        norm_diff=True,
        out_node_nf=None,
        tanh=False,
        coords_range=15,
        agg="sum",
    ):
        super().__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range) / self.n_layers
        if agg == "mean":
            self.coords_range_layer = self.coords_range_layer * 19
        # Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        # self.embedding = TimeEmbedding(self.hidden_nf) # add a time embed to DM
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=self.coords_range_layer,
                    agg=agg,
                ),
            )

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        # h = h.squeeze(-1) #  for Fourier Embedding
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](
                h,
                edges,
                x,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.

    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
        recurrent=True,
        attention=False,
        clamp=False,
        norm_diff=True,
        tanh=False,
        coords_range=1,
        agg="sum",
    ):
        super().__init__()
        input_edge = input_nf * 2
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.agg_type = agg
        self.tanh = tanh
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = coords_range

        self.coord_mlp = nn.Sequential(*coord_mlp)

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_cross_mlp = []
        coord_cross_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_cross_mlp.append(act_fn)
        coord_cross_mlp.append(layer)
        if self.tanh:
            coord_cross_mlp.append(nn.Tanh())
            self.coords_cross_range = coords_range

        self.coord_cross_mlp = nn.Sequential(*coord_cross_mlp)
        self.clamp = clamp

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        # if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)

    def edge_model(self, source, target, radial, edge_attr, edge_mask):
        # print("edge_model", radial, edge_attr)
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        if edge_mask is not None:
            out = out * edge_mask
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        # print("node_model", edge_attr)
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask):
        # print("coord_model", coord_diff, radial, edge_feat)
        row, col = edge_index
        coord_cross = torch.cross(coord[row], coord[col])
        norm = torch.sum((coord_cross) ** 2, 1).unsqueeze(1)
        norm = torch.sqrt(norm + 1e-8)
        coord_cross = coord_cross / (norm + 1)
        if self.tanh:
            trans = coord_diff * self.coord_mlp(edge_feat) * self.coords_range + self.coord_cross_mlp(edge_feat) * self.coords_cross_range * coord_cross
        else:
            trans = coord_diff * self.coord_mlp(edge_feat) + self.coord_cross_mlp(edge_feat) * coord_cross
        # trans = torch.clamp(trans, min=-100, max=100)
        if edge_mask is not None:
            trans = trans * edge_mask

        if self.agg_type == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.agg_type == "mean":
            if node_mask is not None:
                # raise Exception('This part must be debugged before use')
                agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
                M = unsorted_segment_sum(node_mask[col], row, num_segments=coord.size(0))
                agg = agg / (M - 1)
            else:
                agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception("Wrong coordinates aggregation type")
        # print("update", coord, coord_diff,edge_feat, self.coord_mlp(edge_feat), self.coords_range, agg, self.tanh)
        coord = coord + agg
        return coord

    def forward(
        self,
        h,
        edge_index,
        coord,
        edge_attr=None,
        node_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, edge_mask)
        coord = self.coord_model(
            coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask
        )

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        # print("h", h)
        if node_mask is not None:
            h = h * node_mask
            coord = coord * node_mask
        return h, coord, edge_attr

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)

        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff / (norm + 1)

        return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)



def remove_mean(samples, n_particles, n_dimensions):
    """Makes a configuration of many particle system mean-free.

    Parameters
    ----------
    samples : torch.Tensor
        Positions of n_particles in n_dimensions.

    Returns
    -------
    samples : torch.Tensor
        Mean-free positions of n_particles in n_dimensions.
    """
    shape = samples.shape
    if isinstance(samples, torch.Tensor):
        samples = samples.view(-1, n_particles, n_dimensions)
        samples = samples - torch.mean(samples, dim=1, keepdim=True)
        samples = samples.view(*shape)
    else:
        samples = samples.reshape(-1, n_particles, n_dimensions)
        samples = samples - samples.mean(axis=1, keepdims=True)
        samples = samples.reshape(*shape)
    return samples