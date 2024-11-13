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


def log_prob(action, mu, log_std):
    std = torch.exp(log_std)
    return (
        -((action - mu)**2) / (2 * std**2) - log_std - math.log(math.sqrt(2 * math.pi))
    )

def square_loss(action, mu, log_std):
    val = (action - mu) 
    norm = torch.inner(val,val)
    return (norm)
    
class BCAgent(object):
    """Behavioral cloning agent with pointclound input using pointet++ encoder"""

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
        print("actor lr: {},encoder_lr: {}".format(actor_lr, encoder_lr))
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
        self.gripper_idx = 2 if args.observation_mode == 'pointcloud_3' else 1

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

        

        self.replay_buffer_num = args.replay_buffer_num

        # one alpha for each region in the context of pcgrad

        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        
        if self.args.resume_from_ckpt:
            self.load(self.args.resume_from_path_actor, load_optimizer=True)

            
        if self.args.lr_decay is not None:
            # Actor is halved due to delayed update
            self.actor_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones=np.arange(15, 150, 15) * 5000, gamma=0.5)

        self.train()

        #iql parameters

        if self.args.constant_reward is not None:
            self.args.constant_reward = float(self.args.constant_reward)
            print("Using constant reward: ", self.args.constant_reward)
        
        self.loss_type = 'negative_log_likelihood'
        if self.loss_type == 'l2':
            self.loss_fn = square_loss
        else:
            self.loss_fn = log_prob

  

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        if self.encoder_type == 'pixel':
            self.CURL.train(training)


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

    def update_actor(self, obs_array, actions_array, L, step, non_randomized_obs_array=None, pose_id=None, sample_buffer_indices=[0]):
        actor_losses = []
        # print("Updating Actor")
        for i, (obs, action, non_randomized_obs, region_idx) in enumerate(zip(obs_array, actions_array, non_randomized_obs_array, sample_buffer_indices)):
            # print(len(obs))
            mu, pi, log_pi, log_std = self.actor(obs, detach_encoder=self.detach_encoder)
            # print(action.shape)
            if 'flow' in self.encoder_type:
                log_pi = log_pi[obs.x[:, self.gripper_idx] == 1]
            # new_action = action.clone()
            for b_idx in range(self.args.batch_size):
                action[(obs.batch == b_idx)] = action[(obs.batch == b_idx) & (obs.x[:, self.gripper_idx] == 1)]
            bc_losses = -self.loss_fn(action,mu,log_std).sum(-1, keepdim=False)
            actor_loss = torch.mean(bc_losses)

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
        obs_array, action_array, next_obs_array, not_done_array, cpc_kwargs_array, auxiliary_kwargs_array,  =   [], [], [], [], [],[]
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
            next_obs_array.append(next_obs)
            not_done_array.append(not_done)
            cpc_kwargs_array.append(cpc_kwargs)
            auxiliary_kwargs_array.append(auxiliary_kwargs)
 
                

        non_randomized_obs_array = [None for _ in obs_array]
        if self.args.full_obs_guide:
            non_randomized_obs_array = [auxiliary_kwargs['non_randomized_obses'] for auxiliary_kwargs in auxiliary_kwargs_array]
        

        if self.args.train_actor:
            self.update_actor(obs_array,action_array, L, step, non_randomized_obs_array, pose_id=pose_id, sample_buffer_indices=sample_buffer_indices)

        del obs_array
        del next_obs_array

    def load_actor_ckpt(self, actor_load_name):
        self.load_helper(self.actor, actor_load_name)
        print("loaded actor model from {}".format(actor_load_name))
        
    def save(self, model_dir, step, episode, is_best_train=False, is_best_test=False, best_avg_return_train=None, best_avg_return_test=None):
        # print("save model to {} {}".format(model_dir, step))
        actor_save_name = '%s/actor_%s.pt' % (model_dir, step)

        if is_best_train:
            actor_save_name = '%s/actor_best_train_%s_%s.pt' % (model_dir, step, best_avg_return_train)
            
        if is_best_test:
            actor_save_name = '%s/actor_best_test_%s_%s.pt' % (model_dir, step, best_avg_return_test)
        actor_opt = self.actor_optimizer.state_dict()

        torch.save(
            {
                'episode': episode,
                'step': step,
                'model_state_dict': self.actor.state_dict(),
                'actor_optimizer_state_dict': actor_opt,
                'best_avg_return_test': best_avg_return_test,
            
            }, actor_save_name
        )
        if self.encoder_type == 'pixel':
            self.save_curl(model_dir, step)


    def save_curl(self, model_dir, step):
        torch.save(
            self.CURL.state_dict(), '%s/curl_%s.pt' % (model_dir, step)
        )

    def load(self, actor_path,load_optimizer=False):
        actor_checkpoint  = torch.load(actor_path)
        
        self.load_helper(self.actor, actor_path)
        

        if load_optimizer:
            self.actor_optimizer.load_state_dict(
                    actor_checkpoint['actor_optimizer_state_dict']
            )
           

        print("loaded actor model from {}".format(actor_path))
       

    def load_helper(self, model, ckpt_path):
        ckpt  = torch.load(osp.join(ckpt_path))
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(
                ckpt['model_state_dict']
            )
        else:
            model.load_state_dict(ckpt)