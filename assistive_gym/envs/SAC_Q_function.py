import copy
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

# import wandb
import utils
from encoder import make_encoder
from pointnet2 import MLP
from torch_geometric.data import Data, Batch
from SAC_AWAC import gaussian_logprob, squash, weight_init


LOG_FREQ = 10000

def expectile_loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

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
      self, obs, force_vector, compute_pi=True, compute_log_pi=True, detach_encoder=False, visual=False
    ):
        indices = None
        if 'mask' not in self.encoder_type:
            if not visual:
                obs, indices = self.encoder(obs, force_vector, detach=detach_encoder, visual=False)
            else:
                obs, indices, visual_out = self.encoder(obs, detach=detach_encoder, visual=True)

        else:
            obs, masked_obs = self.encoder(obs, detach=detach_encoder, visual=False)
            visual_out = masked_obs
        # if indices is not None:
        #     self.set_maxpool_indices(indices)

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


class QFunction(nn.Module):
    """MLP for q-function."""

    def __init__(self, obs_dim, action_dim, force_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim + force_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action, force_history):
        assert obs.size(0) == action.size(0) and action.size(0) == force_history.size(0)

        obs_action_forcehistory = torch.cat([obs, action, force_history], dim=1)
        return self.trunk(obs_action_forcehistory)


class FlowCritic(nn.Module):
    def __init__(
      self, obs_shape, action_shape, force_history_shape, force_direction_history_shape, action_history_shape, ee_pose_shape, hidden_dim, encoder_type,
      encoder_feature_dim, num_layers, num_filters,
      args,
    ):
        super().__init__()

        self.encoder_type = encoder_type

        new_args = copy.deepcopy(args)

        self.add_force_and_action_to_latent_feature = args.__dict__.get('add_force_and_action_to_latent_feature', False)

        if args.exclude_garment and args.observation_mode == 'pointcloud_3':
            new_args.pc_feature_dim -= 1

        if not self.add_force_and_action_to_latent_feature:
            new_args.pc_feature_dim += action_shape[0] + force_history_shape[0] + force_direction_history_shape[0] + action_history_shape[0] + ee_pose_shape[0]

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, 
            new_args,
            output_logits=True
        )

        feature_dim = self.encoder.feature_dim
        if self.add_force_and_action_to_latent_feature:
            feature_dim += action_shape[0] + force_history_shape[0] + force_direction_history_shape[0] + action_history_shape[0] + ee_pose_shape[0]
        
        label_dim = 1

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, label_dim)
        )
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, label_dim)
        )

        self.outputs = dict()
        self.apply(weight_init)
        self.args = args
        self.maxpool_indices = None

    def forward(self, obs, action, force_history=None, force_direction_history=None, action_history=None, ee_pose=None, detach_encoder=False):
        indices = None
        new_obs = obs.clone()

        if not self.add_force_and_action_to_latent_feature:
            new_obs.x = torch.cat([new_obs.x, action], dim=-1)
            if force_history is not None:
                new_obs.x = torch.cat([new_obs.x, force_history], dim=-1)
            if force_direction_history is not None:
                new_obs.x = torch.cat([new_obs.x, force_direction_history], dim=-1)
            if action_history is not None:
                new_obs.x = torch.cat([new_obs.x, action_history], dim=-1)


        obs, indices = self.encoder(new_obs, detach=detach_encoder)
        if self.add_force_and_action_to_latent_feature:
            # print(obs.shape, action.shape, force_history.shape, action_history.shape, force_direction_history.shape)
            obs = torch.cat([obs, action], dim=-1)
            if force_history is not None:
                obs = torch.cat([obs, force_history], dim=-1)
            if force_direction_history is not None:
                obs = torch.cat([obs, force_direction_history], dim=-1)
            if action_history is not None:
                obs = torch.cat([obs, action_history], dim=-1)
            if ee_pose is not None:
                obs = torch.cat([obs, ee_pose], dim=-1)

        q1 = self.Q1(obs)
        q2 = self.Q2(obs)

        return q1, q2


class ValueFunction(nn.Module):
    def __init__(
      self, obs_shape, force_history_shape, hidden_dim, encoder_type,
      encoder_feature_dim, num_layers, num_filters,
      args,
    ):
        super().__init__()

        self.encoder_type = encoder_type

        new_args = copy.deepcopy(args)
        new_args.pc_feature_dim += force_history_shape[0]
        
        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, 
            new_args,
            output_logits=True
        )

        self.value = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.outputs = dict()
        self.apply(weight_init)
        self.args = args

    def forward(self, obs, force_history=None, detach_encoder=False):
        indices = None
        new_obs = obs.clone()
        new_obs.x = torch.cat([new_obs.x, force_history], dim=-1)
        obs, indices = self.encoder(new_obs, detach=detach_encoder)
        value = self.value(obs)
        return value


def weighted_mse_loss(input, target, weight):
    # print(weight.shape, target.shape)
    # print(weight)
    return torch.sum(weight * (input - target) ** 2)


class ForceVisionAgent(object):
    """CURL representation learning with SAC."""

    def __init__(
      self,
      obs_shape,
      action_shape,
      force_history_shape,
      force_direction_history_shape,
      action_history_shape,
      ee_pose_shape,
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
      encoder_type='pointcloud',
      encoder_feature_dim=50,
      encoder_tau=0.005,
      num_layers=4,
      num_filters=32,
      cpc_update_freq=1,
      log_interval=100,
      detach_encoder=False,
      curl_latent_dim=128,
      actor_load_name=None,
      vision_critic_load_name=None,
      force_critic_load_name=None,
      force_dynamics_load_name=None,
      beta=2,
      agent='curl_sac',
      parallel_gpu=False,
      mode='supervised',
      **kwargs
    ):

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

        self.alpha_fixed = alpha_fixed
        self.beta = beta
        self.parallel_gpu = parallel_gpu
        self.mode = mode

        self.pc_encoder_names = ['pointcloud', 'pointcloud_flow', 'dgcnn', 'dgcnn_flow', 'pt', 'pt_flow']

        self.gripper_idx = args.gripper_idx

        actor_class = Actor

        self.actor = actor_class(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters,
            args
        ).to(device)

        if actor_load_name is not None:
            self.load_helper(self.actor, actor_load_name)
            print("loaded actor model from {}".format(actor_load_name))

        critic_encoder_type = encoder_type
        if ('flow' in self.encoder_type):
            critic_encoder_type = encoder_type[:-5]

        self.force_critic = FlowCritic(
            obs_shape, action_shape, force_history_shape, force_direction_history_shape, action_history_shape, ee_pose_shape, hidden_dim, critic_encoder_type,
            encoder_feature_dim, num_layers, num_filters,
            args,
        ).to(device)

        # print(self.force_critic)
        if force_critic_load_name is not None:
            self.load_helper(self.force_critic, force_critic_load_name)
            print("loaded force critic model from {}".format(force_critic_load_name))

        if args.__dict__.get('plan_mode') == 'constrain_both_within_thres':
            self.force_dynamics = FlowCritic(
                obs_shape, action_shape, force_history_shape, hidden_dim, critic_encoder_type,
                encoder_feature_dim, num_layers, num_filters,
                args,
            ).to(device)

            if force_dynamics_load_name is not None:
                self.load_helper(self.force_dynamics, force_dynamics_load_name)
                print("loaded force dynamics model from {}".format(force_dynamics_load_name))

        self.force_critic_target = FlowCritic(
            obs_shape, action_shape, force_history_shape, force_direction_history_shape, action_history_shape, ee_pose_shape, hidden_dim, critic_encoder_type,
            encoder_feature_dim, num_layers, num_filters,
            args,
        ).to(device)

        self.force_critic_target.load_state_dict(self.force_critic.state_dict())

        self.vision_critic = FlowCritic(
            obs_shape, action_shape, [0], [0], [0], [0], hidden_dim, critic_encoder_type,
            encoder_feature_dim, num_layers, num_filters,
            args,
        ).to(device)

        if vision_critic_load_name is not None:
            self.load_helper(self.vision_critic, vision_critic_load_name)
            print("loaded vision critic model from {}".format(vision_critic_load_name))

        if self.args.use_iql:
            self.force_value_function = ValueFunction(
                obs_shape, force_history_shape, hidden_dim, critic_encoder_type,
                encoder_feature_dim, num_layers, num_filters,
                args,
            ).to(device)
            self.expectile = self.args.expectile


        self.log_alpha = torch.log(init_temperature * torch.ones(1, device=device))    
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        self.add_force_and_action_to_latent_feature = args.__dict__.get('add_force_and_action_to_latent_feature', False)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.force_critic_optimizer = torch.optim.Adam(
            self.force_critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        if self.args.use_iql:
            self.force_value_function_optimizer = torch.optim.Adam(
                self.force_value_function.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
            )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.args.resume_from_ckpt:
            self.load(self.args.resume_from_path_actor, self.args.resume_from_path_critic, load_optimizer=True)

        if self.args.lr_decay is not None:
            # Actor is halved due to delayed update
            self.actor_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones=np.arange(15, 150, 15) * 5000, gamma=0.5)
            self.force_critic_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.force_critic_optimizer, milestones=np.arange(15, 150, 15) * 10000, gamma=0.5)

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.device = device

        self.train()
        self.force_critic_target.train()


    def train(self, training=True):
        self.evaluate = False
        self.training = training
        # self.actor.train(training)
        self.force_critic.train(training)
        self.force_critic_target.train(training)

    def eval(self):
        self.evaluate = True
        self.force_critic.eval()
        self.force_critic_target.eval()


    def select_action(self, obs, force_vector):
        with torch.no_grad():
            obs_ = copy.deepcopy(obs)
            obs_.batch = torch.from_numpy(np.zeros(obs.x.shape[0], dtype=np.int))
            obs_ = obs_.to(self.device)
            force_vector = force_vector.to(self.device)
            mu, _, _, _ = self.actor(
                obs_, force_vector, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()


    def plan_action_constrain_force(self, obs, force_vector, force_history):
        plan_mode = self.args.plan_mode

        with torch.no_grad():
            obs_ = copy.deepcopy(obs)
            obs_.batch = torch.from_numpy(np.zeros(obs.x.shape[0], dtype=np.int))
            obs_ = obs_.to(self.device)
            force_vector = force_vector.to(self.device)
            mu, _, _, log_std = self.actor(
                obs_, force_vector, compute_pi=False, compute_log_pi=False
            )
            mu = mu.cpu().data.numpy().flatten()
            mu = mu.reshape(-1, 6)[-1].flatten()
            log_std = log_std.cpu().data.numpy().flatten()
            log_std = log_std.reshape(-1, 6)[-1].flatten()
            std = np.exp(log_std)
            cov = np.diag(std) * 0.1
            batched_obs = [obs] * self.args.batch_size
            batched_obs = Batch.from_data_list(batched_obs).to(self.device)

            force_history = torch.from_numpy(force_history.astype(np.float32))
            batched_force_history = force_history.unsqueeze(0).repeat(self.args.batch_size,1).to(self.device)

            global_max_vision_idx = None
            global_max_vision_prob = -1e5
            final_action = mu
            predicted_force = 0

            next_mean = copy.deepcopy(mu)
            next_std = copy.deepcopy(std)

            for iter in range(3):
                noise = np.random.normal(size=(self.args.batch_size, 6))
                action_sample = np.tile(next_mean, (self.args.batch_size,1)) + noise * next_std 
                action_sample = action_sample.astype(np.float32)

                if self.args.random_exploration:
                    for i in range(self.args.batch_size):
                        p = np.random.uniform(0, 1)

                        if not self.args.use_cem:
                            exp_prob = self.args.explore_prob
                        else:
                            exp_prob = max(self.args.explore_prob - 0.1 * iter, 0)

                        if p < exp_prob:
                            rand_action = np.random.uniform(-1, 1, 6)
                            action_sample[i] = rand_action
                            noise[i] = (action_sample[i] - mu) / std

                    if plan_mode == 'constrain_force_within_thres':
                        action_sample[-1] = mu
                        noise[-1] = (action_sample[-1] - mu) / std
                        
                batched_action_sample = np.repeat(action_sample, repeats=obs.x.shape[0], axis=0)
                batched_action_sample = torch.from_numpy(batched_action_sample).to(self.device)

                force_q_value_1, force_q_value_2 = self.get_Q_value('force', self.force_critic, batched_obs, batched_action_sample, batched_force_history)
                force_q_value = torch.min(force_q_value_1, force_q_value_2)
                vision_log_pi = gaussian_logprob(torch.from_numpy(noise).to(self.device), torch.from_numpy(log_std).to(self.device))

                if plan_mode == "constrain_force_within_thres":
                    force_q_value += torch.sum(batched_force_history, dim=1)[:, None]
                    constraint_satistying_idx = force_q_value.flatten() < self.args.constrain_force_threshold
                    constraint_satistying_idx = constraint_satistying_idx.nonzero().flatten()
                    if len(constraint_satistying_idx) == 0:
                        print("no samples satisfying constraints, default to action with minumum force")
                        _, constraint_satistying_idx = torch.topk(-force_q_value.flatten(), 1)
                
                constraint_satistying_idx = constraint_satistying_idx.cpu().detach().numpy()

                if self.args.use_cem:
                    next_mean = np.mean(action_sample[constraint_satistying_idx], axis=0)
                    next_std = np.std(action_sample[constraint_satistying_idx], axis=0)
                    
                max_vision_prob = torch.max(vision_log_pi[constraint_satistying_idx])
                max_vision_idx = torch.argmax(vision_log_pi[constraint_satistying_idx])
                if max_vision_prob > global_max_vision_prob:
                    global_max_vision_prob = max_vision_prob
                    global_max_vision_idx = max_vision_idx
                    final_action = action_sample[constraint_satistying_idx[global_max_vision_idx]]
                    predicted_force = force_q_value[constraint_satistying_idx[global_max_vision_idx]].cpu().detach().numpy()[0]

            # print('predicted: ', predicted_force)
            return final_action, mu, predicted_force


    def sample_action(self, obs, force_vector, return_std=False):
        with torch.no_grad():
            obs_ = copy.deepcopy(obs)
            obs_.batch = torch.from_numpy(np.zeros(obs.x.shape[0], dtype=np.int))
            obs_ = obs_.to(self.device)
            force_vector = force_vector.to(self.device)
            mu, pi, _, logstd = self.actor(obs_, force_vector, compute_log_pi=False)
            if not return_std:
                return pi.cpu().data.numpy().flatten()
            else:
                return pi.cpu().data.numpy().flatten(), logstd.cpu().data.numpy().flatten()


    def get_Q_value(self, type, critic, obs, action, force_history=None, force_direction_history=None, action_history=None, ee_pose=None, randomize_action=False, randomized_obs=None):
        assert type in ['vision', 'force']
        if not self.add_force_and_action_to_latent_feature:
            if randomized_obs is None:
                new_action = action.clone()
                if type == 'force':
                    new_force_history = torch.zeros(size=[action.shape[0], force_history.shape[1]])
                    new_force_history = new_force_history.to(action.device)
                for b_idx in range(min(obs.batch[-1]+1, self.args.batch_size)):
                    new_action[(obs.batch == b_idx)] = action[(obs.batch == b_idx) & (obs.x[:, self.gripper_idx] == 1)]
                    if randomize_action:
                        p = np.random.uniform(0, 1)
                        if p < self.args.explore_prob:
                            rand_action = np.random.uniform(-1, 1, 6)
                            rand_action = torch.from_numpy(rand_action.astype(np.float32)).to(action.device)
                            new_action[(obs.batch == b_idx)] = rand_action
                    if type == 'force':
                        new_force_history[(obs.batch == b_idx)] = force_history[b_idx]
            else:
                new_action = []
                for b_idx in range(min(obs.batch[-1]+1, self.args.batch_size)):
                    repeat_times = torch.sum(obs.batch == b_idx).item()
                    new_action.append(action[(randomized_obs.batch == b_idx) & (randomized_obs.x[:, self.gripper_idx] == 1)].repeat(repeat_times, 1))
                new_action = torch.cat(new_action, dim=0)

        else:
            new_action = torch.zeros(min(obs.batch[-1]+1, self.args.batch_size), action.shape[1]).to(action.device)
            for b_idx in range(min(obs.batch[-1]+1, self.args.batch_size)):
                new_action[b_idx] = action[(obs.batch == b_idx) & (obs.x[:, self.gripper_idx] == 1)]
            new_force_history = force_history
        
        if type == 'force':
            Q1, Q2 = critic(obs, new_action, force_history=new_force_history, force_direction_history=force_direction_history, action_history=action_history, ee_pose=ee_pose)
        elif type == 'vision':
            Q1, Q2 = critic(obs, new_action, force_history=None)
            
        return Q1, Q2


    def get_force_sum(self, obs, force_pred_dense):
        assert(force_pred_dense.shape[0] == obs.x.shape[0])
        force_pred_sum = torch.zeros(self.args.batch_size, 1).to(self.device)
        for b_idx in range(min(obs.batch[-1]+1, self.args.batch_size)):
            force_pred_sum[b_idx] = torch.sum(force_pred_dense[(obs.batch == b_idx)])
        return force_pred_sum

    def update_batch(self, data, idx):
        obs, next_obs, others, _ = data
        obs = obs.to(self.device)
        next_obs = next_obs.to(self.device)
        others = others.to(self.device)

        force_history, force_direction_history, ee_pose, action, action_history, n_step_return, future_return, next_force_history, not_done = others.force_history, others.force_direction_history, others.ee_pose, others.action, others.action_history, others.n_step_return, others.future_return, others.next_force_history, torch.logical_not(others.done)
        force_history = force_history.to(self.device)
        ee_pose = ee_pose.to(self.device)
        force_direction_history = force_direction_history.to(self.device)
        action_history = action_history.to(self.device)

        if not self.args.add_force_direction:
            force_direction_history = None

        if not self.args.add_action_history:
            action_history=None

        if not self.args.add_gripper_pose:
            ee_pose = None

        next_force_history = next_force_history.to(self.device)
        
        n_step_return = torch.unsqueeze(n_step_return, 1).to(self.device)
        future_return = torch.unsqueeze(future_return, 1).to(self.device)

        not_done = torch.unsqueeze(not_done, 1)

        if self.args.no_bootstrapping:
            if self.args.use_n_step_return:
                target_Q = n_step_return
            else:
                target_Q = future_return
        else:
            with torch.no_grad():
                _, next_action, log_pi, _ = self.actor(next_obs)
                target_reward_Q1, target_reward_Q2 = self.get_Q_value('force', self.force_critic_target, next_obs, next_action, next_force_history, randomize_action=self.args.random_exploration)
                target_reward_dense = torch.min(target_reward_Q1, target_reward_Q2)
                if self.args.dense_pred:
                    target_reward = self.get_force_sum(next_obs, target_reward_dense)
                else:
                    target_reward = target_reward_dense

                target_Q = n_step_return + (not_done * self.discount ** (self.args.n_step_) * target_reward)


        current_Q1_dense, current_Q2_dense = self.get_Q_value('force', self.force_critic, obs, action, force_history, force_direction_history, action_history, ee_pose)     
        if self.args.dense_pred:
            current_Q1 = self.get_force_sum(obs, current_Q1_dense)
            current_Q2 = self.get_force_sum(obs, current_Q2_dense)
        else:
            current_Q1, current_Q2 = current_Q1_dense, current_Q2_dense
        
        if not self.args.use_weighted_loss:
            criterion = nn.MSELoss()
            critic_loss = criterion(current_Q1.float(), target_Q.float()) + criterion(current_Q2.float(), target_Q.float())
        else:
            force_weight = others.force_weight
            force_weight = torch.unsqueeze(force_weight, 1).to(self.device)
            critic_loss = weighted_mse_loss(current_Q1, target_Q, force_weight) + weighted_mse_loss(current_Q2, target_Q, force_weight)

        loss_no_bootstrapping_l1 = 0
        # this is for evaluation only
        with torch.no_grad():
            eval_loss = nn.L1Loss()

            if not self.args.no_bootstrapping:
                loss_bootstrapping = min(eval_loss(current_Q1, target_Q), eval_loss(current_Q2, target_Q)) / torch.mean(torch.abs(target_Q)) * 100
            else:
                loss_bootstrapping = torch.Tensor([0]).to(self.device)

            if self.args.use_n_step_return:
                if not self.args.use_n_step_delta:
                    loss_no_bootstrapping = min(eval_loss(current_Q1, n_step_return), eval_loss(current_Q2, n_step_return)) / torch.mean(torch.abs(n_step_return)) * 100
                else:
                    prev_force_sum = torch.sum(force_history, dim=1)
                    loss_no_bootstrapping = min(eval_loss(current_Q1 + prev_force_sum, n_step_return + prev_force_sum), \
                        eval_loss(current_Q2 + prev_force_sum, n_step_return + prev_force_sum)) / torch.mean(torch.abs(n_step_return + prev_force_sum)) * 100
                    loss_no_bootstrapping_l1 = min(eval_loss(current_Q1 + prev_force_sum, n_step_return + prev_force_sum), \
                        eval_loss(current_Q2 + prev_force_sum, n_step_return + prev_force_sum))
            else:
                loss_no_bootstrapping = min(eval_loss(current_Q1, future_return), eval_loss(current_Q2, future_return)) / torch.mean(torch.abs(future_return)) * 100

        # Optimize the critic
        self.force_critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.args.DQN_Clipping:
            for param in self.force_critic.parameters():
                param.grad.data.clamp_(-1, 1)
        elif self.args.PER_Clipping:
            torch.nn.utils.clip_grad.clip_grad_norm_(self.force_critic.parameters(), 10)
        self.force_critic_optimizer.step()

        if idx % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.force_critic.Q1, self.force_critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.force_critic.Q2, self.force_critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.force_critic.encoder, self.force_critic_target.encoder,
                self.encoder_tau
            )
        return loss_bootstrapping, loss_no_bootstrapping, loss_no_bootstrapping_l1


    def eval_batch(self, data):
        obs, next_obs, others, _ = data
        obs = obs.to(self.device)
        next_obs = next_obs.to(self.device)
        others = others.to(self.device)

        force_history, action, n_step_return = others.force_history, others.action, others.n_step_return
        force_history = force_history.to(self.device)

        if not self.args.add_force_direction:
            force_direction_history = None

        if not self.args.add_action_history:
            action_history=None

        if not self.args.add_gripper_pose:
            ee_pose = None

        n_step_return = torch.unsqueeze(n_step_return, 1).to(self.device)

        with torch.no_grad():
            if self.args.no_bootstrapping:
                if self.args.use_n_step_return:
                    target_Q = n_step_return
                # else:
                #     target_Q = future_return
            # else:
            #     _, next_action, log_pi, _ = self.actor(next_obs)
            #     target_reward_Q1, target_reward_Q2 = self.get_Q_value('force', self.force_critic_target, next_obs, next_action, next_force_history, randomize_action=self.args.random_exploration)
            #     target_reward_dense = torch.min(target_reward_Q1, target_reward_Q2)
            #     if self.args.dense_pred:
            #         target_reward = self.get_force_sum(next_obs, target_reward_dense)
            #     else:
            #         target_reward = target_reward_dense
            #     target_Q = n_step_return + (not_done * self.discount ** (self.args.n_step_) * target_reward)

            current_Q1_dense, current_Q2_dense = self.get_Q_value('force', self.force_critic, obs, action, force_history, force_direction_history, action_history, ee_pose)  
            if self.args.dense_pred:
                current_Q1 = self.get_force_sum(obs, current_Q1_dense)
                current_Q2 = self.get_force_sum(obs, current_Q2_dense)
            else:
                current_Q1, current_Q2 = current_Q1_dense, current_Q2_dense

            eval_loss = nn.L1Loss()
            loss_no_bootstrapping_l1 = 0
            if not self.args.no_bootstrapping:
                loss_bootstrapping = min(eval_loss(current_Q1, target_Q), eval_loss(current_Q2, target_Q)) / torch.mean(torch.abs(target_Q)) * 100
            else:
                loss_bootstrapping = torch.Tensor([0]).to(self.device)
            if self.args.use_n_step_return:
                if not self.args.use_n_step_delta:
                    loss_no_bootstrapping = min(eval_loss(current_Q1, n_step_return), eval_loss(current_Q2, n_step_return)) / torch.mean(torch.abs(n_step_return)) * 100
                else:
                    prev_force_sum = torch.sum(force_history, dim=1)
                    loss_no_bootstrapping = min(eval_loss(current_Q1 + prev_force_sum, n_step_return + prev_force_sum), \
                        eval_loss(current_Q2 + prev_force_sum, n_step_return + prev_force_sum)) / torch.mean(torch.abs(n_step_return + prev_force_sum)) * 100
                    loss_no_bootstrapping_l1 = min(eval_loss(current_Q1 + prev_force_sum, n_step_return + prev_force_sum), \
                        eval_loss(current_Q2 + prev_force_sum, n_step_return + prev_force_sum))
            else:
                loss_no_bootstrapping = min(eval_loss(current_Q1, future_return), eval_loss(current_Q2, future_return)) / torch.mean(torch.abs(future_return)) * 100

        return loss_bootstrapping, loss_no_bootstrapping, loss_no_bootstrapping_l1
    

    def load_helper(self, model, ckpt_path):
        ckpt  = torch.load(osp.join(ckpt_path), map_location=self.device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(
                ckpt['model_state_dict']
            )
        else:
            model.load_state_dict(ckpt)

    def save(self, model_dir, step):
        critic_save_name = '%s/force_critic_%s.pt' % (model_dir, step+1)

        torch.save(
            {
                'step': step,
                'model_state_dict': self.force_critic.state_dict(),
                'critic_optimizer_state_dict': self.force_critic_optimizer.state_dict(),
            }, critic_save_name
        )