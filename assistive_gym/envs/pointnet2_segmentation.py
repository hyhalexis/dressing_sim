import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate
import torch.nn as nn

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import torch_geometric.transforms as T
try:
    from torch_geometric.nn import PointConv, fps, radius, global_max_pool
except:
    from torch_geometric.nn import PointNetConv as PointConv
    from torch_geometric.nn import fps, radius, global_max_pool

import numpy as np

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip

class Net(torch.nn.Module):
    def __init__(self, 
            out_dim=3, 
            num_layer=3, 
            feature_num=4, 
            sa_radius=[0.2, 0.4],
            sa_ratio=[0.2, 0.25],  
            sa_mlp_list=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
            fp_mlp_list=[[256, 256], [256, 128], [128, 128, 128]],
            linear_mlp_list=[128, 128],
            fp_k=[1, 3, 3],
            use_batch_norm=False,
            residual_learning=False, # if we are learning just a residual upon a nominal thing
            use_film = False,
            use_force_hist = False,
            freeze_weights = False,
            freeze_encoder_only = False
        ):
        super(Net, self).__init__()

        # Input channels account for both `pos` and node features.
        self.use_batch_norm = use_batch_norm
        self.num_layer = num_layer
        self.residual_learning = residual_learning

        self.sa_module_list = nn.ModuleList()
        self.sa_module_list.append(SAModule(sa_ratio[0], sa_radius[0], MLP([feature_num + 3, *sa_mlp_list[0]], batch_norm=self.use_batch_norm)))
        for l_idx in range(1, self.num_layer - 1):
            self.sa_module_list.append(SAModule(sa_ratio[l_idx], sa_radius[l_idx], MLP([sa_mlp_list[l_idx - 1][-1] + 3, *sa_mlp_list[l_idx]], batch_norm=self.use_batch_norm)))
        
        self.sa_module_list.append(GlobalSAModule(MLP([sa_mlp_list[self.num_layer - 2][-1] + 3, *sa_mlp_list[self.num_layer - 1]], batch_norm=self.use_batch_norm)))
            
        self.fp_module_list = nn.ModuleList()
        self.fp_module_list.append(FPModule(fp_k[0], MLP([sa_mlp_list[self.num_layer-1][-1] + sa_mlp_list[self.num_layer-2][-1], *fp_mlp_list[0]], batch_norm=self.use_batch_norm)))
        for l_idx in range(self.num_layer-2, 0, -1):
            f_idx = self.num_layer - 1 - l_idx
            self.fp_module_list.append(FPModule(fp_k[f_idx], MLP([fp_mlp_list[f_idx-1][-1] + sa_mlp_list[l_idx-1][-1], *fp_mlp_list[f_idx]], batch_norm=self.use_batch_norm)))
        self.fp_module_list.append(FPModule(fp_k[self.num_layer-1], MLP([sa_mlp_list[0][-1] + feature_num, *fp_mlp_list[self.num_layer-1]], batch_norm=self.use_batch_norm)))

        self.lin_layers = nn.ModuleList()
        in_size = fp_mlp_list[-1][-1]
        for size in linear_mlp_list:
            self.lin_layers.append(torch.nn.Linear(in_size, size))
            in_size = size

        if not self.residual_learning:
            self.out_layer = torch.nn.Linear(in_size, out_dim)
        else:
            # print("pn++ encoder, residual learning!")
            # self.out_layer = torch.nn.Linear(in_size, out_dim)
            # self.out_w = torch.nn.parameter.Parameter(torch.ones((in_size, out_dim)) * 0.001)
            self.out_w = torch.nn.parameter.Parameter(torch.zeros((in_size, out_dim)))


        self.sa_latent_dim = sa_mlp_list[-1][-1]
        self.use_film = use_film
        self.use_force_hist = use_force_hist

        if self.use_film:
            print('FILMMMMMM!!!!!')
            if self.use_force_hist:
                input_dim = 9
            else:
                input_dim = 3
                
            self.film_layers = nn.ModuleList()
            self.film_layers.append(nn.Linear(input_dim, 2*sa_mlp_list[self.num_layer-1][-1]))
            for l_idx in range(self.num_layer):
                self.film_layers.append(nn.Linear(input_dim, 2*fp_mlp_list[l_idx][-1]))

        if freeze_encoder_only:
            for name, param in self.named_parameters():
                if 'sa_module_list' in name:  # Freeze encoder layers
                        param.requires_grad = True
                        print(name)
        
        if freeze_weights:
            for name, param in self.named_parameters():
                if 'film_layers' not in name:  # Freeze encoder layers
                        param.requires_grad = False
                        print(name)

        # self.film_net = nn.Linear(3, 2*sa_mlp_list[self.num_layer-1][-1])

    def get_sa_latent(self, x, pos, batch):
        sa_out = (x, pos, batch)
        for i in range(self.num_layer):
            sa_out = self.sa_module_list[i](*sa_out)

        return sa_out[0]

    def forward(self, x, pos, batch, force_vector, visual=False):
        sa_out = (x, pos, batch)
        sa_outs = [sa_out]
        for i in range(self.num_layer):
            sa_out = self.sa_module_list[i](*sa_out)
            if i == self.num_layer - 1:
                x, pos, batch, indices = sa_out
                if self.use_film:
                    # print(f"force_vector dtype: {force_vector.dtype}")
                    # print(f"film_net dtype: {self.film_layers[0].weight.dtype}")
                    # print('force', force_vector.shape)
                    force_vector = force_vector.squeeze(1)
                    film_out = self.film_layers[0](force_vector.to(torch.float32))
                    # print('out', film_out.shape)
                    gammas, betas = torch.split(film_out, int(film_out.shape[1]/2), dim=1)

                    # print('gammas', gammas.shape)
                    # print('x', x.shape)
                    # print('pos', x.shape)

                    gammas = gammas.mean(dim=0).unsqueeze(0).expand(x.shape[0], -1)
                    # print('gammas2', gammas.shape)
                    betas = betas.mean(dim=0).unsqueeze(0).expand(x.shape[0], -1)
                    x = gammas * x + betas
                    # print('x2', x.shape)
                sa_out =  x, pos, batch
            else:
                x, pos, batch = sa_out
            sa_outs.append(sa_out)

        fp_out = self.fp_module_list[0](*sa_outs[-1], *sa_outs[-2])

        if self.use_film:
            x, pos_skip, batch_skip = fp_out
            film_out = self.film_layers[1](force_vector.to(torch.float32))
            gammas, betas = torch.split(film_out, int(film_out.shape[1]/2), dim=1)
            gammas = gammas.mean(dim=0).unsqueeze(0).expand(x.shape[0], -1)
            betas = betas.mean(dim=0).unsqueeze(0).expand(x.shape[0], -1)
            x = gammas * x + betas
            fp_out = x, pos_skip, batch_skip

        fp_outs = [fp_out]
        for i in range(1, self.num_layer):
            fp_out = self.fp_module_list[i](*fp_out, *sa_outs[-(i+2)])

            if self.use_film:
                x, pos_skip, batch_skip = fp_out
                film_out = self.film_layers[i+1](force_vector.to(torch.float32))
                gammas, betas = torch.split(film_out, int(film_out.shape[1]/2), dim=1)
                gammas = gammas.mean(dim=0).unsqueeze(0).expand(x.shape[0], -1)
                betas = betas.mean(dim=0).unsqueeze(0).expand(x.shape[0], -1)
                x = gammas * x + betas
                fp_out = x, pos_skip, batch_skip

            fp_outs.append(fp_out)

        x, _, _ = fp_out

        for layer in self.lin_layers:
            x = F.relu(layer(x))
            if self.use_batch_norm:
                x = F.dropout(x, p=0.5, training=self.training)
        
        if not self.residual_learning:
            x = self.out_layer(x)
        else:
            # print("residual learning!")
            x = x @ self.out_w

        if visual:
            return x, [sa[0] for sa in sa_outs[1:-1]] + [fp[0] for fp in fp_outs], indices
        else:
            return x, indices

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        if self.ratio < 1:
            idx = fps(pos, batch, ratio=self.ratio)
        else:
            idx = torch.from_numpy(np.arange(batch.shape[0])).to(x.device).long()
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv((x, x[idx]), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        output = global_max_pool(x, batch)
        # x = output[0]
        # indices = output[1]
        x = output
        indices = None
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch, indices


def MLP(channels, batch_norm=True):
    if batch_norm:
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
            for i in range(1, len(channels))
        ])
    else:
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i]), ReLU())
            for i in range(1, len(channels))
        ])