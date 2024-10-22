import torch
import torch.nn as nn
from pointnet2 import RegressionNet
from pointnet2_segmentation import Net
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

# for 100 x 100 inputs
OUT_DIM_100 = {4: 43}

# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32,output_logits=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        if obs_shape[-1] == 64:
            out_dim =OUT_DIM_64[num_layers]
        elif obs_shape[-1] == 84:
            out_dim = OUT_DIM[num_layers]
        elif obs_shape[-1] == 100:
            out_dim = OUT_DIM_100[num_layers]
        else:
            raise NotImplementedError
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()
        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False, visual=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


class PcEncoder(nn.Module):
    """PointNet++ Encoder"""
    def __init__(self, 
        feature_dim=3, 
        num_layer=3,
        sa_radius=[0.2, 0.4],
        sa_ratio=[0.2, 0.25],  
        sa_mlp_list=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
        linear_mlp_list=[512, 256],
        output_dim=32,
        use_batch_norm=False,
        output_logits=False
    ):
        super().__init__()
        self.output_logits = output_logits
        self.pointnet2 = RegressionNet(
            feature_dim, num_layer, sa_radius, sa_ratio, sa_mlp_list, linear_mlp_list, output_dim, use_batch_norm
        )
        self.feature_dim = output_dim
    
    def copy_conv_weights_from(self, source):
        pass

    def forward(self, obs, detach=False, visual=False):
        # assert obs.shape[1] == 3 + self.feature_dim + 1 
        # xyz = obs[:, :3]
        # featuer = obs[:, 3:3+self.feature_dim]
        # batch = obs[:, -1].long()
        out, indices = self.pointnet2.forward(obs.x, obs.pos, obs.batch)
        if detach:
            out = out.detach()
        if not self.output_logits:
            out = torch.tanh(out)
        return out, indices

    def log(self, L, step, log_freq):
        pass


class PcFlowEncoder(nn.Module):
    """PointNet++ segmentation type Encoder"""
    def __init__(self, 
        feature_dim=3, 
        num_layer=3,
        sa_radius=[0.2, 0.4],
        sa_ratio=[0.2, 0.25],  
        sa_mlp_list=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
        linear_mlp_list=[512, 256],
        output_dim=32,
        fp_mlp_list=[[256, 256], [256, 128], [128, 128, 128]],
        fp_k=[1, 3, 3],
        use_batch_norm=False,
        output_logits=False,
        residual=False,
    ):
        super().__init__()
        self.output_logits = output_logits
        self.feature_dim = output_dim
        self.pointnet2 = Net(
            output_dim, 
            num_layer, 
            feature_dim, 
            sa_radius,
            sa_ratio,  
            sa_mlp_list,
            fp_mlp_list,
            linear_mlp_list,
            fp_k,
            use_batch_norm,
            residual
        )
        self.feature_dim = output_dim
    
    def copy_conv_weights_from(self, source):
        pass

    def forward(self, obs, detach=False, visual=False):
        # assert obs.shape[1] == 3 + self.feature_dim + 1 
        # xyz = obs[:, :3]
        # featuer = obs[:, 3:3+self.feature_dim]
        # batch = obs[:, -1].long()
        if not visual:
            out, indices = self.pointnet2.forward(obs.x, obs.pos, obs.batch, visual=False)
        else:
            out, visual, indices = self.pointnet2.forward(obs.x, obs.pos, obs.batch, visual=True)
        if detach:
            out = out.detach()
        if not self.output_logits:
            out = torch.tanh(out)
        if not visual:
            return out, indices
        else:
            return out, indices, visual

    def log(self, L, step, log_freq):
        pass

_AVAILABLE_ENCODERS = {
    'pixel': PixelEncoder, 'identity': IdentityEncoder, 
    # 'pointcloud': PcEncoder,
    'pointcloud_flow': PcFlowEncoder,
}

def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, args,
    output_logits=False, residual=False
):  
    # assert encoder_type in _AVAILABLE_ENCODERS
    if encoder_type in ['pixel', 'identity']:
        return _AVAILABLE_ENCODERS[encoder_type](
            obs_shape, feature_dim, num_layers, num_filters, output_logits
        )
    elif encoder_type == 'pointcloud':
        return PcEncoder(
            args.pc_feature_dim, args.pc_num_layers, args.sa_radius, 
            args.sa_ratio, args.sa_mlp_list, args.linear_mlp_list, 
            feature_dim, 
            False,
            output_logits
        )
    elif encoder_type == 'pointcloud_flow':
        return PcFlowEncoder(
            args.pc_feature_dim, args.pc_num_layers, args.sa_radius, 
            args.sa_ratio, args.sa_mlp_list, args.linear_mlp_list, 
            feature_dim, 
            args.fp_mlp_list, 
            args.fp_k,
            False, 
            output_logits,
            residual
        )

if __name__ == '__main__':
    import numpy as np

    pc_feature_dim = 3
    pc_num_layers = 3
    sa_radius = [0.05, 0.1]
    sa_ratio = [0.4, 0.5]
    sa_mlp_list = [[64, 64, 128], [128, 128, 256], [256, 512, 1024]]
    linear_mlp_list = [128, 128]
    feature_dim = 50
    fp_mlp_list = [[256, 256], [256, 128], [128, 128, 128]]
    fp_k = [1, 3, 3]
    output_logits = False
    residual = False


    encoder = PcFlowEncoder(
        pc_feature_dim, pc_num_layers, sa_radius, 
        sa_ratio, sa_mlp_list, linear_mlp_list, 
        feature_dim, 
        fp_mlp_list, 
        fp_k,
        False, 
        output_logits,
        residual
    )

    diffs = []
    for _ in range(100):
        input_pos = torch.rand(100, 3) * 10
        input_features = torch.rand(100, 3)
        input_batch = torch.zeros(100, dtype=torch.long)
        input_data = Data(x=input_features, pos=input_pos, batch=input_batch)
        output1 = encoder(input_data)

        input_pos_shifted = input_pos + torch.rand(1, 3) * 10
        input_data_shifted = Data(x=input_features, pos=input_pos_shifted, batch=input_batch)
        output2 = encoder(input_data_shifted)
        # print(output1.shape, output2.shape)

        diff = output1 - output2
        diffs.append(torch.sum(diff ** 2).item())

    print("mean diff: ", np.mean(diffs))