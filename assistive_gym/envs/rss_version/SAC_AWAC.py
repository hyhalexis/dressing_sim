import copy
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import utils
from encoder import make_encoder
from rss_version.models.pcgrad_optim import PCGradOptim

LOG_FREQ = 10000

def get_optimizer_stats(optim):
    state_dict = optim.state_dict()['state']
    if len(state_dict) ==0:
        flattened_exp_avg = flattened_exp_avg_sq = 0.
        print('Warning: optimizer dict is empty!')
    else:
        # flattened_exp_avg = flattened_exp_avg_sq = 0.
        flattened_exp_avg = torch.cat([x['exp_avg'].flatten() for x in state_dict.values()]).cpu().numpy()
        flattened_exp_avg_sq = torch.cat([x['exp_avg_sq'].flatten() for x in state_dict.values()]).cpu().numpy()
    return {
        'exp_avg_mean': np.mean(flattened_exp_avg),
        'exp_avg_std': np.std(flattened_exp_avg),
        'exp_avg_min': np.min(flattened_exp_avg),
        'exp_avg_max': np.max(flattened_exp_avg),
        'exp_avg_sq_mean': np.mean(flattened_exp_avg_sq),
        'exp_avg_sq_std': np.std(flattened_exp_avg_sq),
        'exp_avg_sq_min': np.min(flattened_exp_avg_sq),
        'exp_avg_sq_max': np.max(flattened_exp_avg_sq),
    }


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class Actor(nn.Module):
    """MLP actor network."""

    def __init__(
      self, obs_shape, action_shape, hidden_dim, encoder_type,
      encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters,
      args,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, 
            args,
            output_logits=True,
            residual=False
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

        self.maxpool_indices = None

    def forward(
      self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False, visual=False
    ):
        indices = None
        if 'mask' not in self.encoder_type:
            if not visual:
                obs, indices = self.encoder(obs, detach=detach_encoder, visual=False)
            else:
                obs, indices, visual_out = self.encoder(obs, detach=detach_encoder, visual=True)

        else:
            obs, masked_obs = self.encoder(obs, detach=detach_encoder, visual=False)
            visual_out = masked_obs
        if indices is not None:
            self.set_maxpool_indices(indices)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
          self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        # self.outputs['mu'] = mu
        # self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        if not visual:
            return mu, pi, log_pi, log_std
        else:
            return mu, pi, log_pi, log_std, visual_out

    def get_logprob(self, obs, action, detach_encoder=False, gripper_idx=2):
        encoding = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(encoding).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
          self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        std = log_std.exp()
        pi = action
        # pi = mu + noise * std
        noise = (pi - mu) / std
        log_pi = gaussian_logprob(noise, log_std)
        mu, pi, log_pi = squash(mu, pi, log_pi)

        if 'flow' not in self.encoder_type:
            return log_pi
        if 'flow' in self.encoder_type:
            return log_pi[obs.x[:, gripper_idx] == 1]

    def forward_from_feature(self, obs, compute_pi=False, compute_log_pi=False):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
          self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        # self.outputs['mu'] = mu
        # self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)

    def set_maxpool_indices(self, indices):
        self.maxpool_indices = indices.cpu().data.numpy()
        self.maxpool_indices = np.unique(self.maxpool_indices)

    def get_maxpool_indices(self, ):
        return self.maxpool_indices


class QFunction(nn.Module):
    """MLP for q-function."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(
      self, obs_shape, action_shape, hidden_dim, encoder_type,
      encoder_feature_dim, num_layers, num_filters,
      args,
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, 
            args,
            output_logits=True
        )

        self.encoder_type = encoder_type

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs, indices = self.encoder(obs, detach=detach_encoder)
        if 'mask' in self.encoder_type:
            obs, mask_one_hot = obs

        # print("after encoder, obs shape: ", obs.shape)
        # print("action shape: ", action.shape)

        # print(obs.shape)
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        # self.outputs['q1'] = q1
        # self.outputs['q2'] = q2

        return q1, q2

    def forward_from_feature(self, feature, action):
        # detach_encoder allows to stop gradient propogation to encoder

        q1 = self.Q1(feature, action)
        q2 = self.Q2(feature, action)

        # self.outputs['q1'] = q1
        # self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        # self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)



class FlowCritic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(
      self, obs_shape, action_shape, hidden_dim, encoder_type,
      encoder_feature_dim, num_layers, num_filters,
      args,
    ):
        super().__init__()

        self.encoder_type = encoder_type

        new_args = copy.deepcopy(args)
        if args.action_mode == 'rotation' and 'flow' in args.encoder_type:
            if not args.dual_arm:
                new_args.pc_feature_dim += 6
            else:
                new_args.pc_feature_dim += 12
        else:
            new_args.pc_feature_dim += action_shape[0]

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, 
            new_args,
            output_logits=True
        )

        self.Q1 = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.Q2 = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.outputs = dict()
        self.apply(weight_init)
        self.args = args
        self.maxpool_indices = None

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        indices = None
        new_obs = obs.clone()
        new_obs.x = torch.cat([new_obs.x, action], dim=-1)
        obs, indices = self.encoder(new_obs, detach=detach_encoder)
        
        if indices is not None:
            self.set_maxpool_indices(indices)

        if 'mask' in self.encoder_type:
            obs, mask_one_hot = obs

        q1 = self.Q1(obs)
        q2 = self.Q2(obs)

        # self.outputs['q1'] = q1
        # self.outputs['q2'] = q2

        return q1, q2

    def forward_from_feature(self, feature, action):
        # detach_encoder allows to stop gradient propogation to encoder

        q1 = self.Q1(feature, action)
        q2 = self.Q2(feature, action)

        # self.outputs['q1'] = q1
        # self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        # self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)

    def set_maxpool_indices(self, indices):
        self.maxpool_indices = indices.cpu().data.numpy()
        self.maxpool_indices = np.unique(self.maxpool_indices)

    def get_maxpool_indices(self, ):
        return self.maxpool_indices

class CURL(nn.Module):
    """
    CURL
    """

    def __init__(self, z_dim, encoder, encoder_target, output_type="continuous", 
            flow=False):
        super(CURL, self).__init__()

        self.encoder = encoder

        self.encoder_target = encoder_target

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type
        self.flow = flow

    def encode(self, x, detach=False, ema=False, gripper_idx=2):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if self.flow:
            z_out = z_out[x.x[:, gripper_idx] == 1] # extract gripper encoding only

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

class reward_predictor(nn.Module):
    def __init__(
      self, obs_shape, hidden_dim, encoder_type,
      encoder_feature_dim, num_layers, num_filters,
      args,
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, 
            args,
            output_logits=True,
            residual=False
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs):
        obs, indices = self.encoder(obs, detach=False, visual=False)

        pred = self.trunk(obs)

        return pred

class SAC_AWACAgent(object):
    """CURL representation learning with SAC."""

    def __init__(
      self,
      obs_shape,
      action_shape,
      device,
      args,
      hidden_dim=256,
      discount=0.99,
      init_temperature=0.01,
      alpha_lr=1e-3,
      alpha_beta=0.9,
      alpha_fixed=True,
      actor_lr=1e-3,
      actor_beta=0.9,
      actor_log_std_min=-10,
      actor_log_std_max=2,
      actor_update_freq=2,
      critic_lr=1e-3,
      critic_beta=0.9,
      critic_tau=0.005,
      critic_target_update_freq=2,
    #   encoder_type='pixel',
      encoder_type='pointcloud',
      encoder_feature_dim=50,
      encoder_lr=1e-4,
      encoder_tau=0.005,
      num_layers=4,
      num_filters=32,
      cpc_update_freq=1,
      log_interval=100,
      detach_encoder=False,
      curl_latent_dim=128,
      actor_load_name=None,
      critic_load_name=None,
      beta=2,
      agent='curl_sac',
      parallel_gpu=False,
      use_curl=False,
      keypoint_num=9,
      use_teacher_ddpg_loss=False,
      use_teacher_q_regression_loss=False,
      use_teacher_bc_loss=False,
      **kwargs
    ):
        print("actor lr: {}, critic_lr: {}, encoder_lr: {}".format(actor_lr, critic_lr, encoder_lr))

        self.args = args
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.agent = agent
        self.use_curl = use_curl
        self.keypoint_num = keypoint_num

        assert self.args.loss_mask == 'gripper_only'
        self.gripper_idx = 1 if self.args.__dict__.get("observe_only_human", False) or self.args.observation_mode == 'real_partial_pc' else 2


        if self.agent == 'awac':
            self.alpha_fixed = True
        else:
            self.alpha_fixed = alpha_fixed
        self.beta = beta
        self.parallel_gpu = parallel_gpu

        self.pc_encoder_names = ['pointcloud', 'pointcloud_flow', 'dgcnn', 'dgcnn_flow', 'pt', 'pt_flow']

        actor_class = Actor
        ## actor encoder type is pointcloud_flow
        self.actor = actor_class(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters,
            # pointnet++ parameters
            args
        ).to(device)

        if actor_load_name is not None:
            self.load_helper(self.actor, actor_load_name)
            print("loaded actor model from {}".format(actor_load_name))


        if ('flow' in self.encoder_type and self.args.flow_q_mode != 'concat_latent') or \
            ('flow' not in self.encoder_type and self.args.vector_q_mode == 'repeat_action'):
            critic_class = FlowCritic 
            if ('flow' in self.encoder_type):
                critic_encoder_type = encoder_type[:-5] # pointcloud or dgcnn or point_transformer
            else:
                critic_encoder_type = encoder_type
        else:
            critic_class = Critic
            if 'flow' in self.encoder_type:
                critic_encoder_type = encoder_type[:-5]
            else:
                critic_encoder_type = encoder_type
        ## critic encoder type is pointcloud
        self.critic = critic_class(
            obs_shape, action_shape, hidden_dim, critic_encoder_type,
            encoder_feature_dim, num_layers, num_filters,
            # pointnet++ parameters
            args,
        ).to(device)

        if critic_load_name is not None:
            self.load_helper(self.critic, critic_load_name)
            print("loaded critic model from {}".format(critic_load_name))

        self.critic_target = critic_class(
            obs_shape, action_shape, hidden_dim, critic_encoder_type,
            encoder_feature_dim, num_layers, num_filters,
            # pointnet++ parameters
            args,
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        if self.encoder_type == 'pixel':
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.use_pcgrad = args.use_pcgrad
        self.replay_buffer_num = args.replay_buffer_num

        # one alpha for each region in the context of pcgrad
        self.log_alpha = torch.log(init_temperature * torch.ones(self.replay_buffer_num,
                                                                    device=device))
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        if self.use_pcgrad:
            self.actor_optimizer  = PCGradOptim(self.actor_optimizer)
            self.critic_optimizer = PCGradOptim(self.critic_optimizer)

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.args.resume_from_ckpt:
            self.load(self.args.resume_from_path_actor, self.args.resume_from_path_critic, load_optimizer=True)

            
        if self.args.lr_decay is not None:
            # Actor is halved due to delayed update
            self.actor_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones=np.arange(15, 150, 15) * 5000, gamma=0.5)
            self.critic_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.critic_optimizer, milestones=np.arange(15, 150, 15) * 10000, gamma=0.5)

        if self.use_curl:
            self.actor_target_encoder = make_encoder(
                self.encoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters, 
                args,
                output_logits=True
            )

            self.actor_target_encoder.load_state_dict(self.actor.encoder.state_dict())
            flow = True if 'flow' in self.encoder_type else False
            self.actor_CURL = CURL(encoder_feature_dim,
                    self.actor.encoder, self.actor_target_encoder, output_type='continuous', flow=flow
                    ).to(self.device)

            self.actor_encoder_optimizer = torch.optim.Adam(
                self.actor.encoder.parameters(), lr=encoder_lr
            )
            self.actor_cpc_optimizer = torch.optim.Adam(
                self.actor_CURL.parameters(), lr=encoder_lr
            )

            self.critic_CURL = CURL(encoder_feature_dim,
                    self.critic.encoder, self.critic_target.encoder, output_type='continuous').to(self.device)

            self.critic_encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )
            self.critic_cpc_optimizer = torch.optim.Adam(
                self.critic_CURL.parameters(), lr=encoder_lr
            )

        if self.encoder_type == 'pixel':
            # not really used
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.CURL = CURL(obs_shape, encoder_feature_dim,
                             self.curl_latent_dim, self.critic.encoder, self.critic_target.encoder, output_type='continuous').to(self.device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            self.cpc_optimizer = torch.optim.Adam(
                self.CURL.parameters(), lr=encoder_lr
            )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.use_teacher_ddpg_loss = use_teacher_ddpg_loss
        self.use_teacher_q_regression_loss = use_teacher_q_regression_loss
        self.use_teacher_bc_loss = use_teacher_bc_loss

        if self.use_teacher_ddpg_loss or self.use_teacher_q_regression_loss:
            assert(args.teacher_critics_path is not None)
            self.teacher_critics = []
            for _ in range(len(args.teacher_critics_path)):
                self.teacher_critics.append(critic_class(
                    obs_shape, action_shape, hidden_dim, critic_encoder_type,
                    encoder_feature_dim, num_layers, num_filters,
                    args,
                ).to(device))

            self.load_teacher_critics(args.teacher_critics_path)
        
        if self.use_teacher_bc_loss:
            assert(args.teacher_actors_path is not None)
            teacher_args = copy.deepcopy(args)
            teacher_args.pc_feature_dim = args.teacher_pc_feature_dim
            self.teacher_actors = []
            for _ in range(len(args.teacher_actors_path)):
                self.teacher_actors.append(actor_class(
                    obs_shape, action_shape, hidden_dim, encoder_type,
                    encoder_feature_dim, actor_log_std_min, actor_log_std_max,
                    num_layers, num_filters,
                    teacher_args
                ).to(device))
            self.load_teacher_actors(args.teacher_actors_path)

        if self.args.train_reward_predictor:
            self.reward_predictor = reward_predictor(
                obs_shape, hidden_dim, encoder_type,
                encoder_feature_dim,
                num_layers, num_filters,
                args
            ).to(device)
            self.reward_predictor_optimizer = torch.optim.Adam(
                self.reward_predictor.parameters(), lr=args.reward_predictor_lr, betas=(actor_beta, 0.999)
            )

        self.train()
        self.critic_target.train()


    def get_maxpool_indices(self, ):
        maxpool_indices = {}
        maxpool_indices['actor'] = self.actor.get_maxpool_indices()
        maxpool_indices['critic'] = self.critic.get_maxpool_indices()
        return maxpool_indices
        
    def load_teacher_critics(self, teacher_critics_path):
        for i in range(len(teacher_critics_path)):
            if osp.exists(teacher_critics_path[i]):
                self.load_helper(self.teacher_critics[i], teacher_critics_path[i])
                print("loaded teacher critic model from {}".format(teacher_critics_path[i]))
            else:
                print("no teacher critic model found at {}".format(teacher_critics_path[i]))
                self.teacher_critics[i] = None

    def load_teacher_actors(self, teacher_actors_path):
        for i in range(len(teacher_actors_path)):
            if osp.exists(teacher_actors_path[i]):
                self.load_helper(self.teacher_actors[i], teacher_actors_path[i])
                print("loaded teacher actor model from {}".format(teacher_actors_path[i]))
            else:
                print("no teacher actor model found at {}".format(teacher_actors_path[i]))
                self.teacher_actors[i] = None
                exit()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == 'pixel':
            self.CURL.train(training)
        if self.args.train_reward_predictor:
            self.reward_predictor.train(training)

    @property
    def alpha(self):
        # NOTE: alpha value does not matter for awac
        return self.log_alpha.exp()

    def select_action(self, obs, requires_grad=False):
        with torch.set_grad_enabled(requires_grad):
            if self.encoder_type in ['pixel', 'identity']:
                if not isinstance(obs, torch.Tensor):
                    obs = torch.from_numpy(obs)
                obs = obs.to(torch.float32).to(self.device)
                obs_ = obs.unsqueeze(0)
            elif self.encoder_type in self.pc_encoder_names:
                # obs includes concatenate of position and feature. should add batch
                obs_ = copy.deepcopy(obs)
                obs_.batch = torch.from_numpy(np.zeros(obs.x.shape[0], dtype=np.int))
                obs_ = obs_.to(self.device)
                if requires_grad:
                    obs_.pos.requires_grad = True
                    obs_.x.requires_grad = True
                    obs_.requires_grad = True

            mu, _, _, _ = self.actor(
                obs_, compute_pi=False, compute_log_pi=False
            )
            if not requires_grad:
                return mu.cpu().data.numpy().flatten()
            else:
                return mu, obs_


    def sample_action(self, obs, return_std=False):
        if self.encoder_type == 'pixel':
            if obs.shape[-1] != self.image_size:
                obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            if self.encoder_type in ['pixel', 'identity']:
                # print("obs is: ", obs)
                if not isinstance(obs, torch.Tensor):
                    obs = torch.from_numpy(obs)
                obs = obs.to(torch.float32).to(self.device)
                obs_ = obs.unsqueeze(0)
            elif self.encoder_type in self.pc_encoder_names:
                # obs includes concatenate of position and feature. should add batch
                obs_ = copy.deepcopy(obs)
                obs_.batch = torch.from_numpy(np.zeros(obs.x.shape[0], dtype=np.int))
                obs_ = obs_.to(self.device)

            mu, pi, _, logstd = self.actor(obs_, compute_log_pi=False)
            if not return_std:
                return pi.cpu().data.numpy().flatten()
            else:
                return pi.cpu().data.numpy().flatten(), logstd.cpu().data.numpy().flatten()

    def get_Q_value(self, critic, obs, action, randomized_obs=None, real_world=False):
        if 'flow' in self.encoder_type and self.args.flow_q_mode == 'repeat_se3':
            new_action = []
            for b_idx in range(self.args.batch_size):
                repeat_times = torch.sum(obs.batch == b_idx).item()
                new_action.append(action[b_idx].repeat(repeat_times, 1))
            new_action = torch.cat(new_action, dim=0)
            Q1, Q2 = critic(obs, new_action)
        if 'flow' in self.encoder_type and self.args.flow_q_mode == 'all_flow':
            # just concat the flow action with the observation.
            # print("correct branch for computing Q value")
            Q1, Q2 = critic(obs, action)
        if 'flow' in self.encoder_type and self.args.flow_q_mode == 'repeat_gripper':
            if not self.args.dual_arm:
                if not real_world:
                    if randomized_obs is None:
                        new_action = action.clone()
                        for b_idx in range(self.args.batch_size):
                            new_action[(obs.batch == b_idx)] = action[(obs.batch == b_idx) & (obs.x[:, self.gripper_idx] == 1)]
                    else:
                        # this is for getting the Q value of full observation Q function, but using action from the randomized observation policy
                        # in this case, obs is the full observation, randomized_obs is the randomized observation, action is associated with the randomized observation
                        # print("this branch!")
                        new_action = []
                        for b_idx in range(self.args.batch_size):
                            repeat_times = torch.sum(obs.batch == b_idx).item()
                            new_action.append(action[(randomized_obs.batch == b_idx) & (randomized_obs.x[:, self.gripper_idx] == 1)].repeat(repeat_times, 1))
                        new_action = torch.cat(new_action, dim=0)
                else:
                    new_action = []
                    action = action.reshape(-1, 6).float()
                    # print("action_shape: ", action.shape)
                    for b_idx in range(self.args.batch_size):
                        repeat_times = torch.sum(obs.batch == b_idx).item()
                        new_action.append(action[b_idx].repeat(repeat_times, 1))
                    new_action = torch.cat(new_action, dim=0)
                    # print("obs.shape: ", obs.x.shape)
                    # print("new_action.shape: ", new_action.shape)
                    
            else:
                new_action = torch.cat([action.clone(), action.clone()], dim=1)
                for b_idx in range(self.args.batch_size):
                    gripper_actions = action[(obs.batch == b_idx) & (obs.x[:, self.gripper_idx] == 1)]
                    gripper_1_action = gripper_actions[0]
                    gripper_2_action = gripper_actions[1]
                    new_action[(obs.batch == b_idx), :6] = gripper_1_action
                    new_action[(obs.batch == b_idx), 6:] = gripper_2_action

            Q1, Q2 = critic(obs, new_action)

        # NOTE: add concat latent Q function for flow policy
        if 'flow' in self.encoder_type and self.args.flow_q_mode == 'concat_latent':
            action = action[obs.x[:, self.gripper_idx] == 1]
            Q1, Q2 = critic(obs, action)

        if 'flow' not in self.encoder_type and self.args.vector_q_mode == 'concat_latent':
            Q1, Q2 = critic(obs, action)

        if 'flow' not in self.encoder_type and self.args.vector_q_mode == 'repeat_action':
            # print("this branch")
            new_action = []
            for b_idx in range(self.args.batch_size):
                repeat_times = torch.sum(obs.batch == b_idx).item()
                new_action.append(action[b_idx].repeat(repeat_times, 1))
            new_action = torch.cat(new_action, dim=0)
            Q1, Q2 = critic(obs, new_action)
            
        return Q1, Q2

    def update_critic(self, obs_array, action_array, reward_array, next_obs_array, not_done_array, L, step, pretrain_critic=False, pose_id=None, sample_buffer_indices=[0]):
        critic_losses = []
        for i, (obs, action, reward, next_obs, not_done, region_idx) in enumerate(zip(obs_array, action_array, reward_array, next_obs_array, not_done_array, sample_buffer_indices)):
            # print(f"update critic buffer {i}")
            with torch.no_grad():
                _, policy_action, log_pi, _ = self.actor(next_obs)

                target_Q1, target_Q2 = self.get_Q_value(self.critic_target, next_obs, policy_action)

                if self.use_teacher_q_regression_loss and self.teacher_critics[pose_id] is not None:
                    target_teacher_Q1, target_teacher_Q2 = self.get_Q_value(self.teacher_critics[pose_id], next_obs, policy_action)
                    
                if self.agent == 'awac':
                    target_V = torch.min(target_Q1,
                                        target_Q2)
                elif self.agent == 'curl_sac':
                    if 'flow' in self.encoder_type and self.args.loss_mask == 'gripper_only':
                        # only extract the last point for each point cloud
                        if not self.args.dual_arm:
                            log_pi = log_pi[next_obs.x[:, self.gripper_idx] == 1]
                        else:
                            log_pi = log_pi[next_obs.x[:, self.gripper_idx] == 1].reshape(-1, 2)
                            log_pi = torch.sum(log_pi, dim=1)

                    real_world = region_idx == -1
                    alpha_idx = region_idx if region_idx != -1 else 0
                    target_V = torch.min(target_Q1,
                                    target_Q2) - self.alpha[alpha_idx].detach() * log_pi
                                    
                target_Q = reward + (not_done * self.discount * target_V)

            # get current Q estimates
            current_Q1, current_Q2 = self.get_Q_value(self.critic, obs, action, real_world=real_world)

            critic_loss = F.mse_loss(current_Q1,
                                    target_Q) + F.mse_loss(current_Q2, target_Q)

            if self.use_teacher_q_regression_loss and self.teacher_critics[pose_id] is not None:
                critic_regression_loss = 1e-5 * (F.mse_loss(target_teacher_Q1,
                                    target_Q1) + F.mse_loss(target_teacher_Q2, target_Q2))
                critic_loss += critic_regression_loss     
            critic_losses.append(critic_loss)
            if step % self.log_interval == 0:
                if not pretrain_critic:
                    L.log('train_critic/loss_{}'.format(i), critic_loss, step)
                    L.log('train_critic/q1_{}'.format(i), torch.mean(current_Q1), step)
                    if self.use_teacher_q_regression_loss and self.teacher_critics[pose_id] is not None:
                        L.log('train_critic/regression_loss', critic_regression_loss, step)
                    # if self.args.use_wandb:
                    #     wandb.log({'train_critic/loss': critic_loss, 'train_critic/q1': torch.mean(current_Q1).item()})
                else:
                    L.log('train_separate_critic/loss', critic_loss, step)
                    L.log('train_separate_critic/q1', torch.mean(current_Q1), step)
                    if self.args.use_wandb:
                        wandb.log({'train_separate_critic/loss': critic_loss, 'train_separate_critic/q1': torch.mean(current_Q1).item()})

        # Optimize the critic
        if not self.use_pcgrad:
            self.critic_optimizer.zero_grad()
            total_critic_loss = sum(critic_losses)
            total_critic_loss.backward()
            self.critic_optimizer.step()
        else:
            self.critic_optimizer.zero_grad()
            self.critic_optimizer.pc_backward(critic_losses)
            self.critic_optimizer.step()

        if self.args.lr_decay is not None:
            self.critic_lr_scheduler.step()
            L.log('train/critic_lr', self.critic_optimizer.param_groups[0]['lr'], step)


        # self.critic.log(L, step)

    def update_actor_and_alpha(self, obs_array, L, step, non_randomized_obs_array=None, pose_id=None, sample_buffer_indices=[0]):
        actor_losses, alpha_losses = [], []
        for i, (obs, non_randomized_obs, region_idx) in enumerate(zip(obs_array, non_randomized_obs_array, sample_buffer_indices)):
            # print(f"update actor buffer {i}")

            mu, pi, log_pi, log_std = self.actor(obs, detach_encoder=self.detach_encoder)
            if 'flow' in self.encoder_type:
                if not self.args.dual_arm:
                    log_pi = log_pi[obs.x[:, self.gripper_idx] == 1]
                else:
                    log_pi = log_pi[obs.x[:, self.gripper_idx] == 1].reshape(-1, 2)
                    log_pi = torch.sum(log_pi, dim=1)

            actor_Q1, actor_Q2 = self.get_Q_value(self.critic, obs, pi)
            actor_Q = torch.min(actor_Q1, actor_Q2)

            alpha_idx = region_idx if region_idx != -1 else 0
            actor_loss = (self.alpha[alpha_idx].detach() * log_pi - actor_Q).mean()


            if self.args.full_obs_guide:
                teacher_obs = non_randomized_obs
            else:
                teacher_obs = obs

            if self.use_teacher_ddpg_loss and self.teacher_critics[pose_id] is not None and region_idx != -1: # -1 indicates real-world dataset, do not use teacher loss
                if self.args.full_obs_guide:
                    actor_Q1_teacher, actor_Q2_teacher = self.get_Q_value(self.teacher_critics[pose_id], teacher_obs, pi, obs)
                else:
                    actor_Q1_teacher, actor_Q2_teacher = self.get_Q_value(self.teacher_critics[pose_id], teacher_obs, pi)

                actor_Q_teacher = torch.min(actor_Q1_teacher, actor_Q2_teacher)
                teacher_ddpg_loss = - actor_Q_teacher.mean()
                actor_loss += teacher_ddpg_loss

            if self.use_teacher_bc_loss and self.teacher_actors[pose_id] is not None and region_idx != -1: # -1 indicates real-world dataset, do not use teacher loss
                with torch.no_grad():
                    teacher_mu, _, _, teacher_log_std = self.teacher_actors[pose_id](teacher_obs, detach_encoder=self.detach_encoder)
                    teacher_mu  = teacher_mu[teacher_obs.x[:, 2] == 1]
                    teacher_log_std = teacher_log_std[teacher_obs.x[:, 2] == 1]
                    teacher_std = teacher_log_std.exp()
                student_mu  = mu[obs.x[:, self.gripper_idx] == 1]
                student_log_std = log_std[obs.x[:, self.gripper_idx] == 1]
                student_std = student_log_std.exp()
                if self.args.distill_translation_only:
                    teacher_bc_loss = torch.sum((teacher_mu[:, :3] - student_mu[:, :3]) ** 2) + torch.sum((torch.sqrt(teacher_std[:, :3]) - torch.sqrt(student_std[:, :3])) ** 2)
                else:
                    teacher_bc_loss = torch.sum((teacher_mu - student_mu) ** 2) + torch.sum((torch.sqrt(teacher_std) - torch.sqrt(student_std)) ** 2)
                teacher_bc_loss *= self.args.bc_loss_weight
                actor_loss += teacher_bc_loss


            actor_losses.append(actor_loss)

            if step % self.log_interval == 0:
                L.log('train_actor/loss_{}'.format(i), actor_loss, step)
                L.log('train_actor/target_entropy', self.target_entropy, step)
                if self.use_teacher_ddpg_loss and self.teacher_critics[pose_id] is not None:
                    L.log('train_actor/teacher_ddpg_loss', teacher_ddpg_loss, step)
                if self.use_teacher_bc_loss and self.teacher_actors[pose_id] is not None:
                    L.log('train_actor/teacher_bc_loss', teacher_bc_loss, step)
                if self.args.use_wandb:
                    wandb.log({"train_actor/loss": actor_loss})

            if not self.alpha_fixed:
                alpha_loss = (self.alpha[alpha_idx] *
                            (-log_pi - self.target_entropy).detach()).mean()
                alpha_losses.append(alpha_loss)
                if step % self.log_interval == 0:
                    L.log('train_alpha/loss_{}'.format(i),  alpha_loss, step)
                    L.log('train_alpha/value', self.alpha[alpha_idx], step)
                    # if self.args.use_wandb:
                    #     wandb.log({"train_alpha/loss": alpha_loss, 'train_alpha/value': self.alpha.item()})


        # optimize the actor
        if not self.use_pcgrad:
            self.actor_optimizer.zero_grad()
            total_actor_loss = sum(actor_losses)
            total_actor_loss.backward()
            self.actor_optimizer.step()
        else:
            print('using pcgrad')
            self.actor_optimizer.zero_grad()
            self.actor_optimizer.pc_backward(actor_losses)
            self.actor_optimizer.step()

        if self.args.lr_decay is not None:
            self.actor_lr_scheduler.step()
            L.log('train/actor_lr', self.actor_optimizer.param_groups[0]['lr'], step)
        
        # optimize alpha 
        if not self.alpha_fixed:
            self.log_alpha_optimizer.zero_grad()
            total_alpha_loss = sum(alpha_losses)
            total_alpha_loss.backward()
            self.log_alpha_optimizer.step()


    def update_actor_awac(self, obs, action, L, step):
        # detach encoder, so we don't update it with the actor loss

        with torch.no_grad():
            _, pi, log_pi, log_std = self.actor(obs, detach_encoder=False)
            actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=False)
            v_pi = torch.min(actor_Q1, actor_Q2)

            beta = self.beta
            q1_old_actions, q2_old_actions = self.critic(obs, action)
            q_old_actions = torch.min(q1_old_actions, q2_old_actions)

            adv_pi = q_old_actions - v_pi
            weights = F.softmax(adv_pi / beta, dim=0).detach()

        policy_logpp = self.actor.get_logprob(obs, action, detach_encoder=self.detach_encoder)

        if 'flow' in self.encoder_type:
            # only extract the last point for each point cloud
            policy_logpp = policy_logpp[obs.x[:, self.gripper_idx] == 1]
                    

        actor_loss = (-policy_logpp * weights).sum()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        # entropy = 0.5 * log_std.shape[1] * \
        #           (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        # if step % self.log_interval == 0:
        #     L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.args.lr_decay is not None:
            self.actor_lr_scheduler.step()
            L.log('train/actor_lr', self.actor_optimizer.param_groups[0]['lr'], step)


        self.actor.log(L, step)

        # if step % self.log_interval == 0:
        #     actor_stats = get_optimizer_stats(self.actor_optimizer)
        #     for key, val in actor_stats.items():
        #         L.log('train/actor_optim/' + key, val, step)

    def update_cpc(self, obs_anchor, obs_pos, cpc_kwargs, L, step):

        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)

        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
        if step % self.log_interval == 0:
            L.log('train/curl_loss', loss, step)

    def update_curl(self, obs_anchor, obs_pos, action_anchor, action_pos, L, step):
        # print("training using curl loss!!")

        curls = [self.actor_CURL, self.critic_CURL]
        optimizers = [(self.actor_encoder_optimizer, self.actor_cpc_optimizer), 
            (self.critic_encoder_optimizer, self.critic_cpc_optimizer)]
        log_strings = ['train/actor_curl_loss', 'train/critic_curl_loss']
        
        # bed = time.time()
        for idx in range(2):
            # print("idx is: ", idx)
            curl = curls[idx]

            if idx == 1:
                obs_anchor.x = torch.cat([obs_anchor.x, action_anchor], dim=-1)
                obs_pos.x = torch.cat([obs_pos.x, action_pos], dim=-1)

            z_a = curl.encode(obs_anchor)
            z_pos = curl.encode(obs_pos, ema=True)
            # print("z_a shape: ", z_a.shape)
            # print("z_pos shape: ", z_pos.shape)

            logits = curl.compute_logits(z_a, z_pos)
            # print("logits shape: ", logits.shape)

            labels = torch.arange(logits.shape[0]).long().to(self.device)
            loss = self.cross_entropy_loss(logits, labels)

            encoder_optimizer, cpc_optimizer = optimizers[idx]
            encoder_optimizer.zero_grad()
            cpc_optimizer.zero_grad()
            loss.backward()

            encoder_optimizer.step()
            cpc_optimizer.step()

            if step % self.log_interval == 0:
                L.log(log_strings[idx], loss, step)
        # end = time.time()
        # print("time used: ", end - bed)

    def update(self, replay_buffers, L, step, pretrain_critic=False, pose_id=None, real_world_pc_buffer=None):
        if self.use_teacher_bc_loss or self.use_teacher_ddpg_loss or self.use_teacher_q_regression_loss:
            assert(pose_id is not None)
        
        obs_array, action_array, reward_array, next_obs_array, not_done_array, cpc_kwargs_array, auxiliary_kwargs_array = [], [], [], [], [], [], []
        if len(replay_buffers) > self.args.sample_replay_buffer_num:
            sample_buffer_indices = np.random.choice(len(replay_buffers), size=self.args.sample_replay_buffer_num, replace=False)
        else:
            sample_buffer_indices = np.arange(len(replay_buffers))
            
        # for idx in sample_buffer_indices:
        # print(sample_buffer_indices)
        if real_world_pc_buffer is not None:
            # print("adding real-world buffer for training!")
            sampled_buffers = [replay_buffers[i] for i in sample_buffer_indices] + [real_world_pc_buffer]
            sample_buffer_indices = [0, -1] # assume we are not using PCGrad when fine-tuning with real-world data
        else:
            sampled_buffers = [replay_buffers[i] for i in sample_buffer_indices]
        for buffer in sampled_buffers:
            obs, action, reward, next_obs, not_done, cpc_kwargs, auxiliary_kwargs = buffer.sample_proprio(
                    self.use_curl, self.args.full_obs_guide)
            obs_array.append(obs)
            action_array.append(action)
            reward_array.append(reward)
            next_obs_array.append(next_obs)
            not_done_array.append(not_done)
            cpc_kwargs_array.append(cpc_kwargs)
            auxiliary_kwargs_array.append(auxiliary_kwargs)
        
        # print('step - {} self.log_interval - {}'.format(step, self.log_interval))
        if step % self.log_interval == 0 and not pretrain_critic:
            # print('log batch reward!')
            batch_reward = np.mean([reward_.mean().cpu().data.numpy() for reward_ in reward_array])
            L.log('train/batch_reward', batch_reward, step)

        start_time = time.time()
        non_randomized_obs_array = [None for _ in obs_array]
        if self.args.full_obs_guide:
            non_randomized_obs_array = [auxiliary_kwargs['non_randomized_obses'] for auxiliary_kwargs in auxiliary_kwargs_array]

        if self.args.train_reward_predictor:
            reward_predictor_target_array = [auxiliary_kwargs['reward_predictor_target'] for auxiliary_kwargs in auxiliary_kwargs_array]
            reward_predictor_obs_array = [auxiliary_kwargs['reward_predictor_obs'] for auxiliary_kwargs in auxiliary_kwargs_array]
            self.update_reward_predictor(reward_predictor_obs_array, reward_predictor_target_array, L, step)

        if self.args.train_critic:
            self.update_critic(obs_array, action_array, reward_array, next_obs_array, not_done_array, L, step, pretrain_critic, pose_id=pose_id, sample_buffer_indices=sample_buffer_indices)

        if self.args.train_actor and step % self.actor_update_freq == 0 and not pretrain_critic:
            start_time = time.time()
            if self.agent == 'curl_sac':
                self.update_actor_and_alpha(obs_array, L, step, non_randomized_obs_array, pose_id=pose_id, sample_buffer_indices=sample_buffer_indices)
            elif self.agent == 'awac':
                self.update_actor_awac(obs_array, action_array, L, step)

            # print('actor update time:', time.time() - start_time)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

            if self.use_curl:
                utils.soft_update_params(
                    self.actor.encoder, self.actor_target_encoder,
                    self.encoder_tau
                )

        # if step % self.cpc_update_freq == 0 and (self.encoder_type == 'pixel'):
        #     start_time = time.time()
        #     obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
        #     self.update_cpc(obs_anchor, obs_pos, cpc_kwargs, L, step) 

        # if step % self.cpc_update_freq == 0 and self.use_curl:
        #     obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
        #     action_anchor, action_pos = cpc_kwargs["action_anchor"], cpc_kwargs["action_pos"]
        #     self.update_curl(obs_anchor, obs_pos, action_anchor, action_pos, L, step) 

        del obs_array
        del next_obs_array
        # torch.cuda.empty_cache()

    def update_reward_predictor(self, obs_array, target_array, L, step):
        # print('update reward predictor')
        for i in range(len(obs_array)):
            obs = obs_array[i]
            target = target_array[i]
            if target is None:
                continue
            self.reward_predictor_optimizer.zero_grad()
            reward_pred = self.reward_predictor(obs)
            # print('reward_pred shape {}, target shape{}'.format(reward_pred.shape, target.shape))
            reward_pred_loss = F.binary_cross_entropy_with_logits(reward_pred, target)
            reward_pred_loss.backward()
            self.reward_predictor_optimizer.step()
            L.log('train_actor/reward_predictor_loss', reward_pred_loss, step)


    def load_actor_ckpt(self, actor_load_name):
        self.load_helper(self.actor, actor_load_name)
        print("loaded actor model from {}".format(actor_load_name))
        

    # def save(self, model_dir, step, is_best_train=False, is_best_test=False, best_avg_return_train=None, best_avg_return_test=None):
    #     # print("save model to {} {}".format(model_dir, step))
    #     actor_save_name = '%s/actor_%s.pt' % (model_dir, step)
    #     critic_save_name = '%s/critic_%s.pt' % (model_dir, step)

    #     if is_best_train:
    #         actor_save_name = '%s/actor_best_train_%s_%s.pt' % (model_dir, step, best_avg_return_train)
    #         critic_save_name = '%s/critic_best_train_%s_%s.pt' % (model_dir, step, best_avg_return_train)
    #     if is_best_test:
    #         actor_save_name = '%s/actor_best_test_%s_%s.pt' % (model_dir, step, best_avg_return_test)
    #         critic_save_name = '%s/critic_best_test_%s_%s.pt' % (model_dir, step, best_avg_return_test)
        
    #     torch.save(
    #         self.actor.state_dict(), actor_save_name
    #     )
    #     torch.save(
    #         self.critic.state_dict(), critic_save_name
    #     )
        
    #     if self.encoder_type == 'pixel':
    #         self.save_curl(model_dir, step)


    def save(self, model_dir, step, episode, is_best_train=False, is_best_test=False, best_avg_return_train=None, best_avg_return_test=None):
        # print("save model to {} {}".format(model_dir, step))
        actor_save_name = '%s/actor_%s.pt' % (model_dir, step)
        critic_save_name = '%s/critic_%s.pt' % (model_dir, step)

        if is_best_train:
            actor_save_name = '%s/actor_best_train_%s_%s.pt' % (model_dir, step, best_avg_return_train)
            critic_save_name = '%s/critic_best_train_%s_%s.pt' % (model_dir, step, best_avg_return_train)
        if is_best_test:
            actor_save_name = '%s/actor_best_test_%s_%s.pt' % (model_dir, step, best_avg_return_test)
            critic_save_name = '%s/critic_best_test_%s_%s.pt' % (model_dir, step, best_avg_return_test)
        
        if self.args.train_reward_predictor:
            reward_predictor_save_name = '%s/reward_predictor_%s.pt' % (model_dir, step)
            torch.save(
                {
                    'episode': episode,
                    'step': step,
                    'model_state_dict': self.reward_predictor.state_dict(),
                    'optimizer_state_dict': self.reward_predictor_optimizer.state_dict(),
                }, reward_predictor_save_name
            )

        if self.use_pcgrad:
            actor_opt = self.actor_optimizer.optimizer.state_dict()
            critic_opt = self.critic_optimizer.optimizer.state_dict()
        else:
            actor_opt = self.actor_optimizer.state_dict()
            critic_opt = self.critic_optimizer.state_dict()

        torch.save(
            {
                'episode': episode,
                'step': step,
                'model_state_dict': self.actor.state_dict(),
                'actor_optimizer_state_dict': actor_opt,
                'log_alpha_optimizer_state_dict': self.log_alpha_optimizer.state_dict(),
                'best_avg_return_test': best_avg_return_test,
            
            }, actor_save_name
        )
        torch.save(
            {
                'episode': episode,
                'step': step,
                'model_state_dict': self.critic.state_dict(),
                'critic_optimizer_state_dict': critic_opt,
                'best_avg_return_test': best_avg_return_test,
            }, critic_save_name
        )
        
        if self.encoder_type == 'pixel':
            self.save_curl(model_dir, step)


    def save_curl(self, model_dir, step):
        torch.save(
            self.CURL.state_dict(), '%s/curl_%s.pt' % (model_dir, step)
        )

    def load(self, actor_path, critic_path, load_optimizer=False):
        actor_checkpoint  = torch.load(actor_path)
        critic_checkpoint = torch.load(critic_path)

        # self.actor.load_state_dict(
        #     actor_checkpoint['model_state_dict']
        # )
        # self.critic.load_state_dict(
        #     critic_checkpoint['model_state_dict']
        # )

        self.load_helper(self.actor, actor_path)
        self.load_helper(self.critic, critic_path)
        

        if load_optimizer:
            self.actor_optimizer.load_state_dict(
                    actor_checkpoint['actor_optimizer_state_dict']
            )
            self.log_alpha_optimizer.load_state_dict(
                    actor_checkpoint['log_alpha_optimizer_state_dict']
            )
            self.critic_optimizer.load_state_dict(
                    critic_checkpoint['critic_optimizer_state_dict']
            )

        # if self.encoder_type == 'pixel':    
        #     self.CURL.load_state_dict(
        #         torch.load('%s/curl_%s.pt' % (model_dir, step))
        #     )
        print("loaded actor model from {}".format(actor_path))
        print("loaded critic model from {}".format(critic_path))

    def load_helper(self, model, ckpt_path):
        ckpt  = torch.load(osp.join(ckpt_path))
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(
                ckpt['model_state_dict']
            )
        else:
            model.load_state_dict(ckpt)