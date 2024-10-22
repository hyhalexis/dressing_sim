import copy
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb
from encoder import make_encoder
import utils

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

# normal SAC actor network
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
                # print('not mask', obs.shape)
            else:
                obs, indices, visual_out = self.encoder(obs, detach=detach_encoder, visual=True)

        else:
            obs, masked_obs = self.encoder(obs, detach=detach_encoder, visual=False)
            visual_out = masked_obs

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        # print('trunk mu', mu.shape)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
          self.log_std_max - self.log_std_min
        ) * (log_std + 1)

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
        # print('squash mu', mu.shape)

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

        # handle the case for a dense transformation policy architecture, where the real action is the last action that corresponds to the gripper point
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

    # def log(self, L, step, log_freq=LOG_FREQ):
    #     if step % log_freq != 0:
    #         return

    #     for k, v in self.outputs.items():
    #         L.log_histogram('train_actor/%s_hist' % k, v, step)

    #     L.log_param('train_actor/fc1', self.trunk[0], step)
    #     L.log_param('train_actor/fc2', self.trunk[2], step)
    #     L.log_param('train_actor/fc3', self.trunk[4], step)


# normal MLP Q function 
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

# normal SAC critic
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

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        return q1, q2

    def forward_from_feature(self, feature, action):
        # detach_encoder allows to stop gradient propogation to encoder

        q1 = self.Q1(feature, action)
        q2 = self.Q2(feature, action)
        return q1, q2

    # def log(self, L, step, log_freq=LOG_FREQ):
    #     if step % log_freq != 0:
    #         return

    #     for k, v in self.outputs.items():
    #         L.log_histogram('train_critic/%s_hist' % k, v, step)

    #     for i in range(3):
    #         L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
    #         L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


# a critic network that takes as point cloud as input, and concats actions as the feature in the point cloud
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
            new_args.pc_feature_dim += 6
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
        
        q1 = self.Q1(obs)
        q2 = self.Q2(obs)
        return q1, q2

    def forward_from_feature(self, feature, action):
        # detach_encoder allows to stop gradient propogation to encoder

        q1 = self.Q1(feature, action)
        q2 = self.Q2(feature, action)
        return q1, q2

    # def log(self, L, step, log_freq=LOG_FREQ):
    #     if step % log_freq != 0:
    #         return
        
    #     for k, v in self.outputs.items():
    #         L.log_histogram('train_critic/%s_hist' % k, v, step)

    #     for i in range(3):
    #         L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
    #         L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)



class SAC_AWACAgent(object):
    """CURL representation learning with SAC."""

    def __init__(
      self,
      obs_shape,
      action_shape,
      device,
      args,
      hidden_dim=1024,
      discount=0.99,
      init_temperature=0.1,
      alpha_lr=0.0001,
      alpha_beta=0.5,
      alpha_fixed=False,
      actor_lr=1e-4,
      actor_beta=0.9,
      actor_log_std_min=-10,
      actor_log_std_max=2,
      actor_update_freq=4,
      critic_lr=1e-4,
      critic_beta=0.9,
      critic_tau=0.01,
      critic_target_update_freq=2,
      encoder_type='pointcloud_flow',
      encoder_feature_dim=50,
      encoder_lr=1e-06,
      encoder_tau=0.05,
      num_layers=4,
      num_filters=32,
      cpc_update_freq=1,
      log_interval=1,
      detach_encoder=False,
      curl_latent_dim=128,
      actor_load_name=None,
      critic_load_name=None,
      agent='curl_sac',
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

        self.gripper_idx = 2 if args.observation_mode == 'pointcloud_3' else 1
        self.alpha_fixed = alpha_fixed

        self.pc_encoder_names = ['pointcloud', 'pointcloud_flow']

        # build SAC actor
        actor_class = Actor
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

        # build SAC critic
        critic_class = FlowCritic 
        if ('flow' in self.encoder_type):
            critic_encoder_type = encoder_type[:-5] # pointcloud or dgcnn or point_transformer
        else:
            critic_encoder_type = encoder_type

        self.critic = critic_class(
            obs_shape, action_shape, hidden_dim, critic_encoder_type,
            encoder_feature_dim, num_layers, num_filters,
            # pointnet++ parameters
            args,
        ).to(device)

        if critic_load_name is not None:
            self.load_helper(self.critic, critic_load_name)
            print("loaded critic model from {}".format(critic_load_name))

        # build critic target in SAC
        self.critic_target = critic_class(
            obs_shape, action_shape, hidden_dim, critic_encoder_type,
            encoder_feature_dim, num_layers, num_filters,
            # pointnet++ parameters
            args,
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

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

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.args.resume_from_ckpt:
            self.load(self.args.resume_from_path_actor, self.args.resume_from_path_critic, load_optimizer=True)

            
        if self.args.lr_decay is not None:
            # Actor is halved due to delayed update
            self.actor_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones=np.arange(15, 150, 15) * 5000, gamma=0.5)
            self.critic_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.critic_optimizer, milestones=np.arange(15, 150, 15) * 10000, gamma=0.5)

        
        if self.args.use_distillation:
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

        self.train()
        self.critic_target.train()

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
            elif self.encoder_type in self.pc_encoder_names: # we are using this branch
                # obs includes concatenate of position and feature. should add batch
                obs_ = copy.deepcopy(obs)
                obs_.batch = torch.from_numpy(np.zeros(obs.x.shape[0], dtype=np.int64))
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
        with torch.no_grad():
            if self.encoder_type in ['pixel', 'identity']:
                if not isinstance(obs, torch.Tensor):
                    obs = torch.from_numpy(obs)
                obs = obs.to(torch.float32).to(self.device)
                obs_ = obs.unsqueeze(0)
            elif self.encoder_type in self.pc_encoder_names: # we are using this branch
                # obs includes concatenate of position and feature. should add batch
                obs_ = copy.deepcopy(obs)
                obs_.batch = torch.from_numpy(np.zeros(obs.x.shape[0], dtype=np.int64))
                obs_ = obs_.to(self.device)

            mu, pi, _, logstd = self.actor(obs_, compute_log_pi=False)
            if not return_std:
                return pi.cpu().data.numpy().flatten()
            else:
                return pi.cpu().data.numpy().flatten(), logstd.cpu().data.numpy().flatten()

    # the way to get Q value is different if we use the dense transformation policy (flow-based policy) or just a classficaiton point cloud policy
    def get_Q_value(self, critic, obs, action):
        if 'flow' in self.encoder_type and self.args.flow_q_mode == 'repeat_gripper':
            new_action = action.clone()
            for b_idx in range(self.args.batch_size):
                new_action[(obs.batch == b_idx)] = action[(obs.batch == b_idx) & (obs.x[:, self.gripper_idx] == 1)]

            Q1, Q2 = critic(obs, new_action)

        if 'flow' not in self.encoder_type and self.args.vector_q_mode == 'repeat_action':
            # print("this branch")
            new_action = []
            for b_idx in range(self.args.batch_size):
                repeat_times = torch.sum(obs.batch == b_idx).item()
                new_action.append(action[b_idx].repeat(repeat_times, 1))
            new_action = torch.cat(new_action, dim=0)
            Q1, Q2 = critic(obs, new_action)
            
        return Q1, Q2

    # train the critic network as in SAC
    def update_critic(self, obs_array, action_array, reward_array, next_obs_array, not_done_array, step, pretrain_critic=False, pose_id=None, sample_buffer_indices=[0]):
        critic_losses = []
        for i, (obs, action, reward, next_obs, not_done, region_idx) in enumerate(zip(obs_array, action_array, reward_array, next_obs_array, not_done_array, sample_buffer_indices)):
            with torch.no_grad():
                _, policy_action, log_pi, _ = self.actor(next_obs)

                target_Q1, target_Q2 = self.get_Q_value(self.critic_target, next_obs, policy_action)

                if 'flow' in self.encoder_type:
                    # only extract the last point for each point cloud that corresponds to the gripper action
                    log_pi = log_pi[next_obs.x[:, self.gripper_idx] == 1]

                alpha_idx = region_idx if region_idx != -1 else 0
                target_V = torch.min(target_Q1,
                                target_Q2) - self.alpha[alpha_idx].detach() * log_pi
                                    
                target_Q = reward + (not_done * self.discount * target_V)

            # get current Q estimates
            current_Q1, current_Q2 = self.get_Q_value(self.critic, obs, action)

            critic_loss = F.mse_loss(current_Q1,
                                    target_Q) + F.mse_loss(current_Q2, target_Q)

            critic_losses.append(critic_loss)
            # if step % self.log_interval == 0:
            #     if not pretrain_critic:
            #         L.log('train_critic/loss_{}'.format(i), critic_loss, step)
            #         L.log('train_critic/q1_{}'.format(i), torch.mean(current_Q1), step)
            #     else:
            #         L.log('train_separate_critic/loss', critic_loss, step)
            #         L.log('train_separate_critic/q1', torch.mean(current_Q1), step)
            #         if self.args.use_wandb:
            #             wandb.log({'train_separate_critic/loss': critic_loss, 'train_separate_critic/q1': torch.mean(current_Q1).item()})

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        total_critic_loss = sum(critic_losses)
        total_critic_loss.backward()
        self.critic_optimizer.step()

        if self.args.lr_decay is not None:
            self.critic_lr_scheduler.step()
            # L.log('train/critic_lr', self.critic_optimizer.param_groups[0]['lr'], step)

    def update_actor_and_alpha(self, obs_array, step, non_randomized_obs_array=None, pose_id=None, sample_buffer_indices=[0]):
        actor_losses, alpha_losses = [], []
        for i, (obs, non_randomized_obs, region_idx) in enumerate(zip(obs_array, non_randomized_obs_array, sample_buffer_indices)):

            mu, pi, log_pi, log_std = self.actor(obs, detach_encoder=self.detach_encoder)
            if 'flow' in self.encoder_type:
                log_pi = log_pi[obs.x[:, self.gripper_idx] == 1]
       
            if not (self.args.use_distillation and self.args.pure_distillation): # compute SAC loss if pure_distillation is F
                actor_Q1, actor_Q2 = self.get_Q_value(self.critic, obs, pi)
                actor_Q = torch.min(actor_Q1, actor_Q2)

                alpha_idx = region_idx if region_idx != -1 else 0
                actor_loss = (self.alpha[alpha_idx].detach() * log_pi - actor_Q).mean()
            else: # skip SAC loss commputation if pure_distillation is T and use_distillation is T
                actor_loss = 0

            if self.args.use_distillation: # compute distill loss if use_distill is T
                if self.args.full_obs_guide:
                    teacher_obs = non_randomized_obs
                else:
                    teacher_obs = obs

                if self.teacher_actors[pose_id] is not None and region_idx != -1: # -1 indicates real-world dataset, do not use teacher loss
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
                else:
                    print("skipping teacher loss!")

            actor_losses.append(actor_loss)

            # if step % self.log_interval == 0:
                # L.log('train_actor/loss_{}'.format(i), actor_loss, step)
                # L.log('train_actor/target_entropy', self.target_entropy, step)
                # if self.args.use_distillation and self.teacher_actors[pose_id] is not None:
                #     L.log('train_actor/teacher_bc_loss', teacher_bc_loss, step)
                # if self.args.use_wandb:
                #     wandb.log({"train_actor/loss": actor_loss})

            if not self.alpha_fixed and not (self.args.use_distillation and self.args.pure_distillation):
                alpha_loss = (self.alpha[alpha_idx] *
                            (-log_pi - self.target_entropy).detach()).mean()
                alpha_losses.append(alpha_loss)
                # if step % self.log_interval == 0:
                    # L.log('train_alpha/loss_{}'.format(i),  alpha_loss, step)
                    # L.log('train_alpha/value', self.alpha[alpha_idx], step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        total_actor_loss = sum(actor_losses)
        total_actor_loss.backward()
        self.actor_optimizer.step()

        if self.args.lr_decay is not None:
            self.actor_lr_scheduler.step()
            # L.log('train/actor_lr', self.actor_optimizer.param_groups[0]['lr'], step)
        
        # optimize alpha 
        if not self.alpha_fixed and not self.args.pure_distillation:
            self.log_alpha_optimizer.zero_grad()
            total_alpha_loss = sum(alpha_losses)
            total_alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffers, step, pretrain_critic=False, pose_id=None, real_world_pc_buffer=None):
        if self.args.use_distillation:
            assert (pose_id is not None)
        
        obs_array, action_array, reward_array, next_obs_array, not_done_array, cpc_kwargs_array, auxiliary_kwargs_array = [], [], [], [], [], [], []
        if len(replay_buffers) > self.args.sample_replay_buffer_num:
            sample_buffer_indices = np.random.choice(len(replay_buffers), size=self.args.sample_replay_buffer_num, replace=False)
        else:
            sample_buffer_indices = np.arange(len(replay_buffers))
            
        sampled_buffers = [replay_buffers[i] for i in sample_buffer_indices]
        for buffer in sampled_buffers:
            obs, action, reward, next_obs, not_done, cpc_kwargs, auxiliary_kwargs = buffer.sample_proprio(
                    False, self.args.full_obs_guide)
            obs_array.append(obs)
            action_array.append(action)
            reward_array.append(reward)
            next_obs_array.append(next_obs)
            not_done_array.append(not_done)
            cpc_kwargs_array.append(cpc_kwargs)
            auxiliary_kwargs_array.append(auxiliary_kwargs)
        
        if step % self.log_interval == 0 and not pretrain_critic:
            batch_reward = np.mean([reward_.mean().cpu().data.numpy() for reward_ in reward_array])
            # L.log('train/batch_reward', batch_reward, step)

        non_randomized_obs_array = [None for _ in obs_array]
        if self.args.full_obs_guide:
            non_randomized_obs_array = [auxiliary_kwargs['non_randomized_obses'] for auxiliary_kwargs in auxiliary_kwargs_array]

        if self.args.train_critic:
            self.update_critic(obs_array, action_array, reward_array, next_obs_array, not_done_array, step, pretrain_critic, pose_id=pose_id, sample_buffer_indices=sample_buffer_indices)

        if self.args.train_actor and step % self.actor_update_freq == 0 and not pretrain_critic:
            self.update_actor_and_alpha(obs_array, step, non_randomized_obs_array, pose_id=pose_id, sample_buffer_indices=sample_buffer_indices)


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

        del obs_array
        del next_obs_array

    def load_actor_ckpt(self, actor_load_name):
        self.load_helper(self.actor, actor_load_name)
        print("loaded actor model from {}".format(actor_load_name))
        
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