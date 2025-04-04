import copy
import os.path as osp
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import utils
from encoder import make_encoder
from SAC_AWAC import *
EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class VNet(nn.Module):
    """Value Network from point clouds"""

    def __init__(
      self, obs_shape, action_shape, args, encoder_type='pointcloud', use_action = False, hidden_dim = 128,
      encoder_feature_dim = 50, num_layers = 4, num_filters = 32, 
    ):
        super().__init__()

        self.encoder_type = encoder_type
        new_args = copy.deepcopy(args)
        if use_action:
            if args.action_mode == 'rotation' and 'flow' in args.encoder_type:
                new_args.pc_feature_dim += 6
            else:
                new_args.pc_feature_dim += action_shape[0]
        encoder_type = 'pointcloud'
        new_args.encoder_type = encoder_type
        self.use_action = use_action
        # print("PC2Reward -- Encoder type: ", encoder_type)
        # print("Critic -- Encoder feature dim: ", encoder_feature_dim)
        # print("Critic -- Num layers: ", num_layers)
        # print("Critic -- Num filters: ", num_filters)
        # print("Critic -- args.feature_dim: ", args.pc_feature_dim)
        # print("Critic -- args.pc_num_layers: ", args.pc_num_layers)
        # print("Critic -- args.sa_radius: ", args.sa_radius)
        # print("Critic -- args.sa_ratio: ", args.sa_ratio)
        # print("Critic -- args.sa_mlp_list: ", args.sa_mlp_list)
        # print("Critic -- args.linear_mlp_list: ", args.linear_mlp_list)
        # print("Critic -- args.fp_mlp_list: ", args.fp_mlp_list)
        # print("Critic -- args.fp_k: ", args.fp_k)    
        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, 
            new_args,
            output_logits=True
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
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
        # print("PC2Reward -- new_obs.x", new_obs.x.shape)
        # print("PC2Reward -- obs batch size", new_obs.batch.size())
        # print("PC2Reward -- obs type ",type(new_obs))
        # print("PC2Reward -- action", action.shape)
        if self.use_action: new_obs.x = torch.cat([new_obs.x, action], dim=-1)
        # print("PC2Reward -- Obs shape after concat: ", new_obs.x.shape)
        # print("PC2Reward -- Obs batch size: ", new_obs)
        obs, indices = self.encoder(new_obs, detach=detach_encoder)
        # print("PC2Reward -- Obs shape after encoder: ", obs.shape)
        reward = self.trunk(obs)
        # print("PC2Reward -- reward length", len(reward))
        return reward

    def forward_from_feature(self, feature, action):
        # detach_encoder allows to stop gradient propogation to encoder

        reward = self.trunk(feature)
        return reward

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return
        
        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)

def log_prob(action, mu, log_std):
    std = torch.exp(log_std)
    return (
        -((action - mu)**2) / (2 * std**2) - log_std - math.log(math.sqrt(2 * math.pi))
    )

class IQLAgent(object):
    """CURL representation learning with SAC."""

    def __init__(
      self,
      obs_shape,
      action_shape,
      device,
      args,
      hidden_dim=256,
      discount=0.99,
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
      value_load_name=None,
      agent='curl_iql',
      iql_tau: float = 0.7,
      beta: float = 3.0,
      max_steps: int = 1000000,
      tau: float = 0.005,
      **kwargs
    ):
        print("actor lr: {}, critic_lr: {}, encoder_lr: {}".format(actor_lr, critic_lr, encoder_lr))
        print("Actor pc _dimension: ", args.pc_feature_dim, "teacher pc_feature dimension: ", args.teacher_pc_feature_dim)
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
        self.encoder_feature_dim = encoder_feature_dim
        self.gripper_idx = args.gripper_idx
        # self.gripper_idx = 2 if args.observation_mode == 'pointcloud_3' else 1

        self.pc_encoder_names = ['pointcloud', 'pointcloud_flow']

        # build SAC actor
        actor_class = Actor
        teacher_args = copy.deepcopy(args)
        # teacher_args.pc_feature_dim = args.teacher_pc_feature_dim #Uncomment this line if you are loading one of the teacher models
        self.actor = actor_class(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters,
            # pointnet++ parameters
            teacher_args
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
        print("critic encoder type -- IQL: ", critic_encoder_type)
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

        self.value_network = VNet(obs_shape, action_shape, hidden_dim=self.args.hidden_dim, encoder_type="pointcloud", 
                                       encoder_feature_dim=self.encoder_feature_dim, num_layers=self.args.num_layers, num_filters=self.args.num_filters, args=self.args).float().to(device)
        if value_load_name is not None:
            self.load_helper(self.value_network, value_load_name)
            print("loaded value model from {}".format(value_load_name))

        self.replay_buffer_num = args.replay_buffer_num

        # one alpha for each region in the context of pcgrad

        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )
        
        if self.args.resume_from_ckpt:
            self.load(self.args.resume_from_path_actor, self.args.resume_from_path_critic, self.args.resume_from_path_value, load_optimizer=True)

            
        if self.args.lr_decay is not None:
            # Actor is halved due to delayed update
            self.actor_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones=np.arange(15, 150, 15) * 5000, gamma=0.5)
            self.critic_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.critic_optimizer, milestones=np.arange(15, 150, 15) * 10000, gamma=0.5)
            self.value_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.value_optimizer, milestones=np.arange(15, 150, 15) * 10000, gamma=0.5)


        self.train()
        self.critic_target.train()

        #iql parameters
        self.iql_tau = iql_tau
        self.beta = beta
        self.tau = tau

        if self.args.constant_reward is not None:
            self.args.constant_reward = float(self.args.constant_reward)
            print("Using constant reward: ", self.args.constant_reward)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.value_network.train(training)
        if self.encoder_type == 'pixel':
            self.CURL.train(training)


    def select_action(self, obs, force_vector, requires_grad=False):
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
            force_vector = force_vector.to(self.device)
            if requires_grad:
                force_vector.requires_grad = True
            mu, _, _, _ = self.actor(
                obs_, force_vector, compute_pi=False, compute_log_pi=False
            )
            if not requires_grad:
                return mu.cpu().data.numpy().flatten()
            else:
                return mu, obs_


    def sample_action(self, obs, force_vector, return_std=False):
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
            force_vector = force_vector.to(self.device)
            mu, pi, _, logstd = self.actor(obs_, force_vector, compute_log_pi=False)
            if not return_std:
                return pi.cpu().data.numpy().flatten()
            else:
                return pi.cpu().data.numpy().flatten(), logstd.cpu().data.numpy().flatten()

    # the way to get Q value is different if we use the dense transformation policy (flow-based policy) or just a classficaiton point cloud policy
    def get_Q_value(self, critic, obs, action):
        # print("Getting Q Value")
        if 'flow' in self.encoder_type and self.args.flow_q_mode == 'repeat_gripper':
           
            new_action = action.clone()
            for b_idx in range(self.args.batch_size):
                new_action[(obs.batch == b_idx)] = action[(obs.batch == b_idx) & (obs.x[:, self.gripper_idx] == 1)]
            # print("-----get_q_value: ",obs.x.shape, new_action.shape)
            # print(len(obs), len(action))
            Q1, Q2 = critic(obs, new_action)
            # print(len(Q1), len(Q2))

        if 'flow' not in self.encoder_type and self.args.vector_q_mode == 'repeat_action':
            # print("this branch")
            new_action = []
            for b_idx in range(self.args.batch_size):
                repeat_times = torch.sum(obs.batch == b_idx).item()
                new_action.append(action[b_idx].repeat(repeat_times, 1))
            new_action = torch.cat(new_action, dim=0)
            Q1, Q2 = critic(obs, new_action)
            
        return Q1, Q2
    
    def update_value_network(self, obs_array, action_array, reward_array, next_obs_array, not_done_array, L, step, pretrain_critic=False, pose_id=None, sample_buffer_indices=[0]):
        value_losses = []
        adv_array = []
        for i, (obs, action, reward, next_obs, not_done, region_idx) in enumerate(zip(obs_array, action_array, reward_array, next_obs_array, not_done_array, sample_buffer_indices)):
            with torch.no_grad():                   
                target_Q = torch.min(*self.get_Q_value(self.critic_target, obs, action))
            
            v = self.value_network(obs, action)
            adv = target_Q - v
            adv_array.append(adv)
            v_loss = asymmetric_l2_loss(adv, self.iql_tau)

            value_losses.append(v_loss)
            if step % self.log_interval == 0:
                if not pretrain_critic:
                    L.log('train_value/loss_{}'.format(i), v_loss, step)
                    L.log('train_critic/q1_{}'.format(i), torch.mean(target_Q), step)
                    if self.args.use_wandb:
                        wandb.log({'train_value/loss': v_loss, 'train_value/q1': torch.mean(target_Q).item()})
                else:
                    L.log('train_value/loss', v_loss, step)
                    L.log('train_value/q1', torch.mean(target_Q), step)
                    if self.args.use_wandb:
                        wandb.log({'train_value/loss': v_loss, 'train_value/q1': torch.mean(target_Q).item()})

        # Optimize the critic
        self.value_optimizer.zero_grad()
        total_value_loss = sum(value_losses)
        total_value_loss.backward()
        self.value_optimizer.step()

        if self.args.lr_decay is not None:
            self.value_lr_scheduler.step()
            L.log('train/critic_lr', self.value_lr_scheduler.param_groups[0]['lr'], step)
        return adv_array
    # train the critic network as in SAC
    def update_critic(self, next_v_array, obs_array, action_array, reward_array, next_obs_array, not_done_array, L, step, pretrain_critic=False, pose_id=None, sample_buffer_indices=[0]):
        critic_losses = []
        # print("Updating Critic")
        for i, (next_v, obs, action, reward, next_obs, not_done, region_idx) in enumerate(zip(next_v_array, obs_array, action_array, reward_array, next_obs_array, not_done_array, sample_buffer_indices)):
            with torch.no_grad():
                terminals = not_done
                targets = reward + (terminals.float()) * self.discount * next_v.detach()


            # get current Q estimates
            current_Q1, current_Q2 = self.get_Q_value(self.critic, obs, action)

            critic_loss = F.mse_loss(current_Q1,
                                    targets) + F.mse_loss(current_Q2, targets)

            critic_losses.append(critic_loss)
            if step % self.log_interval == 0:
                if not pretrain_critic:
                    L.log('train_critic/loss_{}'.format(i), critic_loss, step)
                    L.log('train_critic/q1_{}'.format(i), torch.mean(current_Q1), step)
                    if self.args.use_wandb:
                        wandb.log({'train_separate_critic/loss': critic_loss, 'train_separate_critic/q1': torch.mean(current_Q1).item()})
                else:
                    L.log('train_separate_critic/loss', critic_loss, step)
                    L.log('train_separate_critic/q1', torch.mean(current_Q1), step)
                    if self.args.use_wandb:
                        wandb.log({'train_separate_critic/loss': critic_loss, 'train_separate_critic/q1': torch.mean(current_Q1).item()})

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        total_critic_loss = sum(critic_losses)
        total_critic_loss.backward()
        self.critic_optimizer.step()

        if self.args.lr_decay is not None:
            self.critic_lr_scheduler.step()
            L.log('train/critic_lr', self.critic_optimizer.param_groups[0]['lr'], step)

    def update_actor(self, adv_array, obs_array, actions_array, force_vector_array, L, step, non_randomized_obs_array=None, pose_id=None, sample_buffer_indices=[0]):
        actor_losses = []
        # print("Updating Actor")
        for i, (adv, obs, action, force_vector, non_randomized_obs, region_idx) in enumerate(zip(adv_array, obs_array, actions_array, force_vector_array, non_randomized_obs_array, sample_buffer_indices)):
            exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
            mu, pi, log_pi, log_std = self.actor(obs, force_vector, detach_encoder=self.detach_encoder)
            if 'flow' in self.encoder_type:
                log_pi = log_pi[obs.x[:, self.gripper_idx] == 1]
            new_action = action.clone()
            # print("update actor ------New action shape: ", new_action.shape)
            for b_idx in range(self.args.batch_size):
                new_action[(obs.batch == b_idx)] = action[(obs.batch == b_idx) & (obs.x[:, self.gripper_idx] == 1)]
            bc_losses = -log_prob(new_action,mu,log_std).sum(-1, keepdim=False)
            actor_loss = torch.mean(exp_adv * bc_losses)

            actor_losses.append(actor_loss)

            if step % self.log_interval == 0:
                L.log('train_actor/loss_{}'.format(i), actor_loss, step)
                L.log('train_actor/target_entropy', self.target_entropy, step)
                if self.args.use_wandb:
                    wandb.log({"train_actor/loss": actor_loss})
                    wandb.log({"train_actor/target_entropy": self.target_entropy})  

        # optimize the actor
        self.actor_optimizer.zero_grad()
        total_actor_loss = sum(actor_losses)
        total_actor_loss.backward()
        self.actor_optimizer.step()

        if self.args.lr_decay is not None:
            self.actor_lr_scheduler.step()
            L.log('train/actor_lr', self.actor_optimizer.param_groups[0]['lr'], step)

    def update(self, replay_buffers, L, step, pretrain_critic=False, pose_id=None, real_world_pc_buffer=None):
        # print("Calling update")
        obs_array, action_array, reward_array, next_obs_array, not_done_array, force_vector_array, cpc_kwargs_array, auxiliary_kwargs_array, next_v_array = [], [], [], [], [], [], [],[], []
        if len(replay_buffers) > self.args.sample_replay_buffer_num:
            sample_buffer_indices = np.random.choice(len(replay_buffers), size=self.args.sample_replay_buffer_num, replace=False)
        else:
            sample_buffer_indices = np.arange(len(replay_buffers))
            
        sampled_buffers = [replay_buffers[i] for i in sample_buffer_indices]
        for buffer in sampled_buffers:
            obs, action, reward, next_obs, not_done, force_vector, cpc_kwargs, auxiliary_kwargs = buffer.sample_proprio(
                    False, self.args.full_obs_guide)
            obs_array.append(obs)
            action_array.append(action)
            reward_array.append(reward)
            next_obs_array.append(next_obs)
            not_done_array.append(not_done)
            force_vector_array.append(force_vector)
            cpc_kwargs_array.append(cpc_kwargs)
            auxiliary_kwargs_array.append(auxiliary_kwargs)
            with torch.no_grad():
                next_v = self.value_network(next_obs, action)
                next_v_array.append(next_v)
            # print("update-----Action shape: ", action.shape)    
            
        if self.args.constant_reward is not None:
            for i in range(len(reward_array)):
                reward_array_buffer = reward_array[i]
                reward_array_buffer = torch.full_like(reward_array_buffer, self.args.constant_reward)
                reward_array[i] = reward_array_buffer
                # print(reward_array_buffer)

        
        if step % self.log_interval == 0 and not pretrain_critic:
            batch_reward = np.mean([reward_.mean().cpu().data.numpy() for reward_ in reward_array])
            L.log('train/batch_reward', batch_reward, step)

        non_randomized_obs_array = [None for _ in obs_array]
        if self.args.full_obs_guide:
            non_randomized_obs_array = [auxiliary_kwargs['non_randomized_obses'] for auxiliary_kwargs in auxiliary_kwargs_array]
        

        if self.args.train_actor:
            adv_array = self.update_value_network(obs_array, action_array, reward_array, next_obs_array, not_done_array, L, step, pretrain_critic, pose_id=pose_id, sample_buffer_indices=sample_buffer_indices)
            self.update_critic(next_v_array, obs_array, action_array, reward_array, next_obs_array, not_done_array, L, step, pretrain_critic, pose_id=pose_id, sample_buffer_indices=sample_buffer_indices)
            self.update_actor(adv_array,obs_array,action_array, force_vector_array, L, step, non_randomized_obs_array, pose_id=pose_id, sample_buffer_indices=sample_buffer_indices)


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
        value_save_name = '%s/value_%s.pt' % (model_dir, step)

        if is_best_train:
            actor_save_name = '%s/actor_best_train_%s_%s.pt' % (model_dir, step, best_avg_return_train)
            critic_save_name = '%s/critic_best_train_%s_%s.pt' % (model_dir, step, best_avg_return_train)
            value_save_name = '%s/value_best_train_%s_%s.pt' % (model_dir, step, best_avg_return_train)
            
        if is_best_test:
            actor_save_name = '%s/actor_best_test_%s_%s.pt' % (model_dir, step, best_avg_return_test)
            critic_save_name = '%s/critic_best_test_%s_%s.pt' % (model_dir, step, best_avg_return_test)
            value_save_name = '%s/value_best_test_%s_%s.pt' % (model_dir, step, best_avg_return_test)
        
        actor_opt = self.actor_optimizer.state_dict()
        critic_opt = self.critic_optimizer.state_dict()
        value_opt = self.value_optimizer.state_dict()

        torch.save(
            {
                'episode': episode,
                'step': step,
                'model_state_dict': self.actor.state_dict(),
                'actor_optimizer_state_dict': actor_opt,
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
        torch.save(
            {
                'episode': episode,
                'step': step,
                'model_state_dict': self.value_network.state_dict(),
                'value_optimizer_state_dict': value_opt,
                'best_avg_return_test': best_avg_return_test,
            }, value_save_name
        )
        
        if self.encoder_type == 'pixel':
            self.save_curl(model_dir, step)


    def save_curl(self, model_dir, step):
        torch.save(
            self.CURL.state_dict(), '%s/curl_%s.pt' % (model_dir, step)
        )

    def load(self, actor_path, critic_path, value_path, load_optimizer=False):
        actor_checkpoint  = torch.load(actor_path, map_location=self.device)
        critic_checkpoint = torch.load(critic_path, map_location=self.device)
        value_checkpoint = torch.load(value_path, map_location=self.device)

        self.load_helper(self.actor, actor_path)
        self.load_helper(self.critic, critic_path)
        self.load_helper(self.value_network, value_path)
        

        if load_optimizer:
            self.actor_optimizer.load_state_dict(
                    actor_checkpoint['actor_optimizer_state_dict']
            )
            self.critic_optimizer.load_state_dict(
                    critic_checkpoint['critic_optimizer_state_dict']
            )
            self.value_optimizer.load_state_dict(
                    value_checkpoint['value_optimizer_state_dict']
            )

        print("loaded actor model from {}".format(actor_path))
        print("loaded critic model from {}".format(critic_path))
        print("loaded value model from {}".format(value_path))

    def load_helper(self, model, ckpt_path):
        ckpt  = torch.load(osp.join(ckpt_path), map_location=self.device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(
                ckpt['model_state_dict'], strict=False
            )
        else:
            model.load_state_dict(ckpt)