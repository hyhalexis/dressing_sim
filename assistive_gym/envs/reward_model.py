import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from torch_geometric.data import Data, Batch
import wandb
import asyncio
from PIL import Image
import datetime
import pickle as pkl
import random
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# from dressing_motion.curl.vlm_reward.prompt import (
#     gemini_free_query_env_prompts, gemini_summary_env_prompts,
#     gemini_free_query_prompt1, gemini_free_query_prompt2,
#     gemini_single_query_env_prompts,
#     gpt_free_query_env_prompts, gpt_summary_env_prompts,
# )
# from dressing_motion.curl.vlm_reward.gemini_infer import gemini_query_2, gemini_query_1
# from conv_net import CNN, fanin_init

# from dressing_motion.curl.pc_replay_buffer import PointCloudReplayBuffer
from encoder import make_encoder
import copy

device = 'cuda:2'
LOG_FREQ = 10000

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net

def gen_image_net(image_height, image_width, 
                  conv_kernel_sizes=[5, 3, 3 ,3], 
                  conv_n_channels=[16, 32, 64, 128], 
                  conv_strides=[3, 2, 2, 2]):
    conv_args=dict( # conv layers
        kernel_sizes=conv_kernel_sizes, # for sweep into, cartpole, drawer open. 
        n_channels=conv_n_channels,
        strides=conv_strides,
        output_size=1,
    )
    conv_kwargs=dict(
        hidden_sizes=[], # linear layers after conv
        batch_norm_conv=False,
        batch_norm_fc=False,
    )

    return CNN(
        **conv_args,
        paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
        input_height=image_height,
        input_width=image_width,
        input_channels=3,
        init_w=1e-3,
        hidden_init=fanin_init,
        **conv_kwargs
    )

def gen_image_net2():
    from torchvision.models.resnet import ResNet
    from torchvision.models.resnet import BasicBlock

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1)
    return model

def KCenterGreedy(obs, full_obs, num_new_sample):
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_smallest_dist(new_obs, new_full_obs)
        max_index = torch.argmax(dist)
        max_index = max_index.item()
        
        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index+1:]
        
        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs, 
            obs[selected_index]], 
            axis=0)
    return selected_index

def compute_smallest_dist(obs, full_obs):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)
                
        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)

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

class PC2Reward(nn.Module):
    """Reward Model which learns preference rewards from point clouds"""

    def __init__(
      self, obs_shape, action_shape, args, use_action = True, hidden_dim = 128, encoder_type='pointcloud',
      encoder_feature_dim = 50, num_layers = 4, num_filters = 32, 
    ):
        super().__init__()

        self.encoder_type = encoder_type
        new_args = copy.deepcopy(args)
        new_args.pc_feature_dim = new_args.reward_pc_feature_dim
        if use_action:
            if args.action_mode == 'rotation' and 'flow' in args.encoder_type:
                new_args.reward_pc_feature_dim += 6
            else:
                new_args.reward_pc_feature_dim += action_shape[0]
        encoder_type = 'pointcloud'
        new_args.encoder_type = encoder_type
        self.use_action = use_action
        # print("PC2Reward -- Encoder type: ", encoder_type)
        # print("Critic -- Encoder feature dim: ", encoder_feature_dim)
        # print("Critic -- Num layers: ", num_layers)
        # print("Critic -- Num filters: ", num_filters)
        # print("Critic -- args.feature_dim: ", args.reward_pc_feature_dim)
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
            nn.Linear(hidden_dim, 1), nn.Tanh()
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

class RewardModel3:
    def __init__(self, ds, da, args,
                 ensemble_size=3, lr=3e-4, mb_size = 512, size_segment=1, 
                 max_size=5000, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0,
                 
                # vlm related params
                vlm_label=True,
                env_name="DressingEnv",
                vlm="gemini_free_form",
                clip_prompt=None,
                log_dir=None,
                flip_vlm_label=False,
                save_query_interval=25,
                cached_label_path=None,
                use_action = True,
                # image based reward
                reward_model_layers=3,
                reward_model_H=256,
                image_reward=True,
                image_height=128,
                image_width=128,
                resize_factor=1,
                resnet=False,
                conv_kernel_sizes=[5, 3, 3 ,3],
                conv_n_channels=[16, 32, 64, 128],
                conv_strides=[3, 2, 2, 2],

                **kwargs
                ):
        
        # train data is trajectories, must process to sa and s..   
        self.ds = ds #state dimension
        self.da = da#action dimension
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        self.args = args
        self.device = device
        print("Reward Model -- Device: ", self.device)
        self.gripper_idx = 2
        self.use_action = use_action
        self.encoder_type = args.encoder_type
        self.encoder_feature_dim = args.encoder_feature_dim

        self.capacity = int(capacity)
        self.reward_model_layers = reward_model_layers
        self.reward_model_H = args.hidden_dim 

        
        self.buffer_seg1 = [_ for _ in range(self.capacity)]
        self.buffer_act1 = [_ for _ in range(self.capacity)]

        self.buffer_seg2 = [_ for _ in range(self.capacity)]
        self.buffer_act2 = [_ for _ in range(self.capacity)]
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
                
        self.construct_ensemble()
        self.inputs = []  # Store observations
        self.actions = []  # Store actions
        self.targets = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 32
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch
        
        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin

        # vlm label
        self.vlm_label = vlm_label
        self.env_name = env_name
        self.vlm = vlm
        self.clip_prompt = clip_prompt
        self.vlm_label_acc = 0
        self.log_dir = log_dir
        self.flip_vlm_label = flip_vlm_label
        self.train_times = 0
        self.save_query_interval = save_query_interval
        
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        
        
        self.read_cache_idx = 0
        if cached_label_path is not None:
            self.cached_label_path = "{}/{}".format(dir_path, cached_label_path)
            all_cached_labels = sorted(os.listdir(self.cached_label_path))
            self.all_cached_labels = [os.path.join(self.cached_label_path, x) for x in all_cached_labels]
        else:
            self.cached_label_path = None
            
        self.debug = False
        
        print("Initialized Reward Model")
    def eval(self,):
        for i in range(self.de):
            self.ensemble[i].eval()

    def train(self,):
        for i in range(self.de):
            self.ensemble[i].train()
    
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
        
    def construct_ensemble(self):
        for i in range(self.de):
            model = PC2Reward(self.ds, self.da, use_action=self.use_action, hidden_dim=self.args.hidden_dim, encoder_type=self.args.encoder_type, encoder_feature_dim=self.encoder_feature_dim, num_layers=self.args.num_layers, num_filters=self.args.num_filters, args=self.args).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
            
    def add_data(self, obs, act, rew, done):

        self.inputs.append(obs)
        self.actions.append(act)
        self.targets.append(rew)
    
    def prepare_test_data(self,):
        for i in tqdm(range(10)):
            self.uniform_sampling()
        self.test_buffer_seg1 = self.buffer_seg1
        self.test_buffer_seg2 = self.buffer_seg2
        self.test_buffer_act1 = self.buffer_act1
        self.test_buffer_act2 = self.buffer_act2
        self.test_buffer_label = self.buffer_label
        self.test_buffer_index = self.buffer_index
        self.test_buffer_full = self.buffer_full
        self.buffer_seg1 = [_ for _ in range(self.capacity)]
        self.buffer_act1 = [_ for _ in range(self.capacity)]
        self.buffer_seg2 = [_ for _ in range(self.capacity)]
        self.buffer_act2 = [_ for _ in range(self.capacity)]
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
        print("test data prepared")
        
    def eval_test_data(self,):
        self.eval()
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.test_buffer_full else self.test_buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = 10 # int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                obs_1 = []
                obs_2 = []
                act_1 = []
                act_2 = []
                for i in range(len(idxs)):
                    obs_1.append(copy.deepcopy(self.test_buffer_seg1[idxs[i]]))
                    obs_2.append(copy.deepcopy(self.test_buffer_seg2[idxs[i]]))
                    act_1.append(copy.deepcopy(self.test_buffer_act1[idxs[i]]))
                    act_2.append(copy.deepcopy(self.test_buffer_act2[idxs[i]]))
                
                if self.debug:
                    print(idxs, "IDX")
                    print(f"train_reward - Member {member} Epoch {epoch} idxs:", idxs)
                    print(f"train_reward - Member {member} Epoch {epoch} obs_1 length:", len(obs_1))
                    print(f"train_reward - Member {member} Epoch {epoch} obs_2 length:", len(obs_2))
                    print(f"train_reward - Member {member} Epoch {epoch} act_1 length:", len(act_1))
                    print(f"train_reward - Member {member} Epoch {epoch} act_2 length:", len(act_2))
                obs_1, act_1, obs_2, act_2 = self.process(obs_1, act_1, obs_2, act_2)
                labels = self.test_buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if self.debug:
                    print(f"train_reward - Member {member} Epoch {epoch} obs_1.x shape:", obs_1.x.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} obs_2.x shape:", obs_2.x.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} act_1 shape:", act_1.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} act_2 shape:", act_2.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} labels shape:", labels.shape)
                
                if member == 0:
                    total += labels.size(0)
                
                # if self.image_reward:
                #     # obs_1 is batch_size x segment x image_height x image_width x 3
                #     obs_1 = np.transpose(obs_1, (0, 1, 4, 2, 3)) # for torch we need to transpose channel first
                #     obs_2 = np.transpose(obs_2, (0, 1, 4, 2, 3)) 
                #     # also we stored uint8 images, we need to convert them to float32
                #     obs_1 = obs_1.astype(np.float32) / 255.0
                #     obs_2 = obs_2.astype(np.float32) / 255.0
                #     obs_1 = obs_1.squeeze(1)
                #     obs_2 = obs_2.squeeze(1)

                # get logits
                r_hat1 = self.r_hat_member(obs_1, act_1, member=member)
                r_hat2 = self.r_hat_member(obs_2, act_2, member=member)
                r_hat1 = r_hat1.sum(axis=1,keepdim=True)
                r_hat2 = r_hat2.sum(axis=1,keepdim=True)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
        
        ensemble_acc = ensemble_acc / total
        if self.args.use_wandb:
            wandb.log({"test-accuracy": np.mean(ensemble_acc)})
        print("test-accuracy:", np.mean(ensemble_acc))
        self.train() 
        return ensemble_acc
          
    def add_data_batch(self, obses, actions, rewards):
        self.inputs.extend(obses)
        self.actions.extend(actions)
        self.targets.extend(rewards)
    
    def process(self,obs_1, act_1, obs_2, act_2, device='cuda'):
        self.device = device
        if self.debug:
            print("process - Size of inputs as obs_1:", len(obs_1))
            print("process - Size of inputs as obs_2:", len(obs_2))
            print("process - Size of inputs as act_1:", len(act_1))
            print("process - Size of inputs as act_2:", len(act_2))
        # obs_1 = copy.deepcopy(obs_1)
        # obs_2 = copy.deepcopy(obs_2)
        # act_1 = copy.deepcopy(act_1)
        # act_2 = copy.deepcopy(act_2)
        if isinstance(obs_1[0], Data):
            # print("process - obs_1[0].x.shape:", obs_1[0].x.shape)
            obs_1 = Batch.from_data_list(obs_1)
            obs_2 = Batch.from_data_list(obs_2)
        else:
            raise ValueError("obs_1 and obs_2 must be lists of Data objects.")
    
        if 'flow' in self.args.encoder_type:
            if self.args.action_mode == 'translation' or (self.args.action_mode == 'rotation' and self.args.flow_q_mode in ['all_flow', 'repeat_gripper', 'concat_latent']):
                act_1 = np.concatenate(act_1)
                act_2 = np.concatenate(act_2)
                act_1 = torch.from_numpy(act_1).float().to(self.device)
                act_2 = torch.from_numpy(act_2).float().to(self.device)
        else:
                act_1 = torch.as_tensor(act_1, device=self.device)
                act_2 = torch.as_tensor(act_2, device=self.device)
        
        if self.debug:
            print("process - Size after making into batch class obs1.x:", obs_1.x.shape)
            print("process - Size after making into batch class obs2.x:", obs_2.x.shape)
            print("process - Size after making into concat act1:", act_1.shape)
            print("process - Size after making into concat act2:", act_2.shape)
            print("process - Batch size of obs1:", obs_1.batch_size)
            print("process - Batch size of obs2:", obs_2.batch_size)
        
        assert obs_1.batch_size == obs_2.batch_size
        for b_idx in range(obs_1.batch_size):
                act_1[(obs_1.batch == b_idx)] = act_1[(obs_1.batch == b_idx) & (obs_1.x[:, self.gripper_idx] == 1)]
                act_2[(obs_2.batch == b_idx)] = act_2[(obs_2.batch == b_idx) & (obs_2.x[:, self.gripper_idx] == 1)]
        self.device = "cuda"
        return obs_1, act_1, obs_2, act_2
    
    def get_rank_probability(self, obs_1, act_1, obs_2, act_2):
        # get probability obs_1, act_1 > obs_2, act_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(obs_1, act_1, obs_2, act_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_entropy(self, obs_1, act_1, obs_2, act_2):
        # get entropy of probability obs_1, act_1 > obs_2, act_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(obs_1, act_1, obs_2, act_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, obs_1, act_1, obs_2, act_2, member=-1):
        # softmaxing to get the probabilities
        with torch.no_grad():
            r_hat1 = self.r_hat_member(obs_1, act_1, member=member)
            r_hat2 = self.r_hat_member(obs_2, act_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        # taking 0 index for probability obs_1, act_1 > obs_2, act_2
        return F.softmax(r_hat, dim=-1)[:,0]
    
    def p_hat_entropy(self, obs_1, act_1, obs_2, act_2, member=-1):
        # softmaxing to get the probabilities
        with torch.no_grad():
            r_hat1 = self.r_hat_member(obs_1, act_1, member=member)
            r_hat2 = self.r_hat_member(obs_2, act_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, obs, act, member=-1):
        # the network parameterizes r hat
        return self.ensemble[member](obs.to(device), act.to(device))

    def r_hat(self, obs, act):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(obs, act, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, obs, act):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(obs, act, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def load(self, model_dir, step):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.join(file_dir, model_dir)
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member), map_location=device)
            )
        print("Finished loading reward model ensemble")
    
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            obs_1 = self.buffer_seg1[epoch*batch_size:last_index]
            act_1 = self.buffer_act1[epoch*batch_size:last_index]
            obs_2 = self.buffer_seg2[epoch*batch_size:last_index]
            act_2 = self.buffer_act2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(obs_1, act_1, member=member)
                r_hat2 = self.r_hat_member(obs_2, act_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
    def get_queries(self, mb_size=20): 
        # We sample a batch of segments and actions from inputs, actions and targets which represent all the data we have
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        max_len = len(self.inputs)
        # get train traj
        train_obses = copy.deepcopy(self.inputs[:max_len])# train_obses = np.array(self.inputs[:max_len])
        train_acts = copy.deepcopy(self.actions[:max_len])# train_acts = np.array(self.actions[:max_len])
        train_targets = copy.deepcopy(self.targets[:max_len])# train_targets = np.array(self.targets[:max_len])
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        obs_2_t = []
        act_2_t = []
        r_t_2_t = []
        for ind in batch_index_2:
            obs_2_t.append(train_obses[ind]) # Batch x T x dim of obs
            act_2_t.append(train_acts[ind]) # Batch x T x dim of act
            r_t_2_t.append(train_targets[ind]) # Batch x T x 1
        
        # print(len(obs_2_t), len(act_2_t), len(r_t_2_t), " LEN of sample 2")
        # print(len(obs_2_t[0]))
        # for i in range(len(obs_2_t)):
        #     print(obs_2_t[i].x.shape, len(act_2_t[i]), r_t_2_t[i], "LEN of each sample")
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        obs_1_t = []
        act_1_t = []
        r_t_1_t = []
        for ind in batch_index_1:
            obs_1_t.append(train_obses[ind]) # Batch x T x dim of obs
            act_1_t.append(train_acts[ind]) # Batch x T x dim of act
            r_t_1_t.append(train_targets[ind]) # Batch x T x 1
        # print(len(obs_1_t), len(act_1_t), len(r_t_1_t), " LEN of sample 1")
        # obs_1 = []
        # act_1 = []
        # r_t_1 = []
        # obs_2 = []
        # act_2 = []
        # r_t_2 = []
        # # Generate time index 
        # for ind in range((len(obs_1_t))):
        #     random_idx_2 = np.random.choice(len_traj-self.size_segment)
        #     obs_2.append(obs_2_t[random_idx_2:random_idx_2+self.size_segment]) # Batch x size_seg x dim of obs
        #     act_2.append(act_2_t[random_idx_2:random_idx_2+self.size_segment]) # Batch x size_seg x dim of act
        #     r_t_2.append(r_t_2_t[random_idx_2:random_idx_2+self.size_segment]) # Batch x size_seg x 1
        #     random_idx_1 = np.random.choice(len_traj-self.size_segment)
        #     obs_1.append(obs_1_t[random_idx_1:random_idx_1+self.size_segment]) # Batch x size_seg x dim of obs
        #     act_1.append(act_1_t[random_idx_1:random_idx_1+self.size_segment]) # Batch x size_seg x dim of act
        #     r_t_1.append(r_t_1_t[random_idx_1:random_idx_1+self.size_segment]) # Batch x size_seg x 1

        # obs_1 = np.array(obs_1)
        # print(act_1[0])
        # act_1 = np.array(act_1_t)
        r_t_1 = np.array(r_t_1_t)
        # obs_2 = np.array(obs_2)
        # act_2 = np.array(act_2_t)
        r_t_2 = np.array(r_t_2_t)
        if self.debug:
            print("get_queries - obs_1_t:", len(obs_1_t), "obs_2_t:", len(obs_2_t), "act_1_t:", len(act_1_t), "act_2_t:", len(act_2_t))
            print("get_queries - obs_1_t[0] shape:", obs_1_t[0].x.shape, "obs_2_t[0] shape:", obs_2_t[0].x.shape, "act_1_t[0] shape:", len(act_1_t[0]), "act_2_t[0] shape:", len(act_2_t[0]))
            print("get_queries - r_t_1 shape:", r_t_1.shape, "r_t_2 shape:", r_t_2.shape)
        return obs_1_t, act_1_t, obs_2_t, act_2_t, r_t_1, r_t_2
    
    def put_queries(self, obs_1, act_1, obs_2, act_2, labels):
        # put queries into the buffer that is bufferseg this is different from the entire dataset. The aim of this buffer is for backpropagation for the reward model
        total_sample = len(obs_1)
        next_index = self.buffer_index + total_sample
        if self.debug:
            print("put_queries - total_sample:", total_sample)
            print("put_queries - next_index:", next_index)
            print("put_queries - buffer_index:", self.buffer_index)
            print("put_queries - obs_1[0] shape:", obs_1[0].x.shape)
            print("put_queries - obs_2[0] shape:", obs_2[0].x.shape)
            print("put_queries - act_1 shape:", len(act_1[0]))
            print("put_queries - act_2 shape:", len(act_2[0]))
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            for i in range(self.buffer_index, self.capacity):
                self.buffer_seg1[i] = obs_1[i]
                self.buffer_seg2[i] = obs_2[i]
                self.buffer_act1[i] = act_1[i]
                self.buffer_act2[i] = act_2[i]
            # np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], obs_1[:maximum_index])
            # np.copyto(self.buffer_act1[self.buffer_index:self.capacity], act_1[:maximum_index])
            # np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], obs_2[:maximum_index])
            # np.copyto(self.buffer_act2[self.buffer_index:self.capacity], act_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                for i in range(remain):
                    self.buffer_seg1[i] = obs_1[maximum_index + i]
                    self.buffer_seg2[i] = obs_2[maximum_index + i]
                    self.buffer_act1[i] = act_1[maximum_index + i]
                    self.buffer_act2[i] = act_2[maximum_index + i]
                # np.copyto(self.buffer_seg1[0:remain], obs_1[maximum_index:])
                # np.copyto(self.buffer_act1[0:remain], act_1[maximum_index:])
                # np.copyto(self.buffer_seg2[0:remain], obs_2[maximum_index:])
                # np.copyto(self.buffer_act2[0:remain], act_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            for i in range(total_sample):
                self.buffer_seg1[self.buffer_index+i] = obs_1[i]
                self.buffer_seg2[self.buffer_index+i] = obs_2[i]
                # print(len(act_1), len(act_2), len(labels), "LEN of act1, act2, labels")
                # print(len(self.buffer_act1), len(self.buffer_act2), len(self.buffer_label), "LEN of buffer act1, act2, labels")
                self.buffer_act1[self.buffer_index+i] = act_1[i]
                self.buffer_act2[self.buffer_index+i] = act_2[i]
            # np.copyto(self.buffer_seg1[self.buffer_index:next_index], obs_1)
            # np.copyto(self.buffer_act1[self.buffer_index:next_index], act_1)
            # np.copyto(self.buffer_seg2[self.buffer_index:next_index], obs_2)
            # np.copyto(self.buffer_act2[self.buffer_index:next_index], act_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index
            
        if self.debug:
            print("put_queries - bufferseg1 shape:", self.buffer_seg1[self.buffer_index-1].x.shape)
            print("put_queries - bufferseg2 shape:", self.buffer_seg2[self.buffer_index-1].x.shape)
            print("put_queries - bufferact1 shape:", len(self.buffer_act1[self.buffer_index-1]))
            print("put_queries - bufferact2 shape:", len(self.buffer_act2[self.buffer_index-1]))
        
            
    def get_label(self, obs_1, act_1, r_t_1, obs_2, act_2, r_t_2):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # # skip the query
        # if self.teacher_thres_skip > 0: 
        #     max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
        #     max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
        #     if sum(max_index) == 0:
        #         return None, None, None, None, None, None, []

        #     obs_1 = obs_1[max_index]
        #     act_1 = act_1[max_index]
        #     r_t_1 = r_t_1[max_index]
        #     obs_2 = obs_2[max_index]
        #     act_2 = act_2[max_index]
        #     r_t_2 = r_t_2[max_index]
        #     sum_r_t_1 = np.sum(r_t_1, axis=1)
        #     sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
            
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                            torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
        
        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]

        # equally preferable
        labels[margin_index] = -1 
        
        # if self.vlm_label:
        #     ts = time.time()
        #     time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

        #     gpt_two_image_paths = []
        #     combined_images_list = []
        #     useful_indices = []
            
        #     file_path = os.path.abspath(__file__)
        #     dir_path = os.path.dirname(file_path)
        #     save_path = "{}/data/gpt_query_image/{}/{}".format(dir_path, self.env_name, time_string)
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
                
        #     for idx, (img1, img2) in enumerate(zip(img_t_1, img_t_2)):
        #         combined_image = np.concatenate([img1, img2], axis=1)
        #         combined_images_list.append(combined_image)
        #         combined_image = Image.fromarray(combined_image)
                
        #         first_image_save_path = os.path.join(save_path, "first_{:06}.png".format(idx))
        #         second_image_save_path = os.path.join(save_path, "second_{:06}.png".format(idx))
        #         Image.fromarray(img1).save(first_image_save_path)
        #         Image.fromarray(img2).save(second_image_save_path)
        #         gpt_two_image_paths.append([first_image_save_path, second_image_save_path])
                

        #         diff = np.linalg.norm(img1 - img2)
        #         if diff < 1e-3: # ignore the pair if the image is exactly the same
        #             useful_indices.append(0)
        #         else:
        #             useful_indices.append(1)
                        
        #     if self.vlm == 'gpt4v_two_image': 
        #         from vlms.gpt4_infer import gpt4v_infer_2
        #         vlm_labels = []
        #         for idx, (img_path_1, img_path_2) in enumerate(gpt_two_image_paths):
        #             print("querying vlm {}/{}".format(idx, len(gpt_two_image_paths)))
        #             query_prompt = gpt_free_query_env_prompts[self.env_name]
        #             summary_prompt = gpt_summary_env_prompts[self.env_name]
        #             res = gpt4v_infer_2(query_prompt, summary_prompt, img_path_1, img_path_2)
        #             try:
        #                 label_res = int(res)
        #             except:
        #                 label_res = -1

        #             vlm_labels.append(label_res)
        #             time.sleep(0.1)
        #     elif self.vlm == 'gemini_single_prompt':
        #         vlm_labels = []
        #         for idx, (img1, img2) in enumerate(zip(img_t_1, img_t_2)):
        #             res = gemini_query_1([
        #                 gemini_free_query_prompt1,
        #                 Image.fromarray(img1), 
        #                 gemini_free_query_prompt2,
        #                 Image.fromarray(img2), 
        #                 gemini_single_query_env_prompts[self.env_name],
        #             ])
        #             try:
        #                 if "-1" in res:
        #                     res = -1
        #                 elif "0" in res:
        #                     res = 0
        #                 elif "1" in res:
        #                     res = 1
        #                 else:
        #                     res = -1
        #             except:
        #                 res = -1 
        #             vlm_labels.append(res)
        #     elif self.vlm == "gemini_free_form":
        #         vlm_labels = []
        #         for idx, (img1, img2) in enumerate(zip(img_t_1, img_t_2)):
        #             res = gemini_query_2(
        #                     [
        #                         gemini_free_query_prompt1,
        #                         Image.fromarray(img1), 
        #                         gemini_free_query_prompt2,
        #                         Image.fromarray(img2), 
        #                         gemini_free_query_env_prompts[self.env_name]
        #             ],
        #                         gemini_summary_env_prompts[self.env_name]
        #             )
        #             try:
        #                 res = int(res)
        #                 if res not in [0, 1, -1]:
        #                     res = -1
        #             except:
        #                 res = -1
        #             vlm_labels.append(res)   

        #     vlm_labels = np.array(vlm_labels).reshape(-1, 1)
            # good_idx = (vlm_labels != -1).flatten()
            # useful_indices = (np.array(useful_indices) == 1).flatten()
            # good_idx = np.logical_and(good_idx, useful_indices)
            
            # obs_1 = obs_1[good_idx]
            # act_1 = act_1[good_idx]
            # r_t_1 = r_t_1[good_idx]
            # obs_2 = obs_2[good_idx]
            # act_2 = act_2[good_idx]
            # r_t_2 = r_t_2[good_idx]
            # rational_labels = rational_labels[good_idx]
            # vlm_labels = vlm_labels[good_idx]
            # combined_images_list = np.array(combined_images_list)[good_idx]
            # img_t_1 = img_t_1[good_idx]
            # img_t_2 = img_t_2[good_idx]
            # if self.flip_vlm_label:
            #     vlm_labels = 1 - vlm_labels

            # if self.train_times % self.save_query_interval == 0 or 'gpt4v' in self.vlm:
            #     save_path = os.path.join(self.log_dir, "vlm_label_set")
            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path)
            #     with open("{}/{}.pkl".format(save_path, time_string), "wb") as f:
            #         pkl.dump([combined_images_list, rational_labels, vlm_labels, obs_1, act_1, obs_2, act_2, r_t_1, r_t_2], f, protocol=pkl.HIGHEST_PROTOCOL)

            # acc = 0
            # if len(vlm_labels) > 0:
            #     acc = np.sum(vlm_labels == rational_labels) / len(vlm_labels)
            #     print("vlm label acc: {}".format(acc))
            #     print("vlm label acc: {}".format(acc))
            #     print("vlm label acc: {}".format(acc))
            # else:
            #     print("no vlm label")
            #     print("no vlm label")
            #     print("no vlm label")

            # self.vlm_label_acc = acc
            # if not self.image_reward:
            #     return obs_1, act_1, obs_2, act_2, r_t_1, r_t_2, labels, vlm_labels
            # else:
            #     return obs_1, act_1, obs_2, act_2, r_t_1, r_t_2, img_t_1, img_t_2, labels, vlm_labels

        # if not self.image_reward:
        labels =  np.array(labels).reshape(-1, 1)
        # print(labels.shape, "LABEL SHAPE")
        return obs_1, act_1, obs_2, act_2, r_t_1, r_t_2, labels
        # else:
        #     return obs_1, act_1, obs_2, act_2, r_t_1, r_t_2, img_t_1, img_t_2, labels
 
    
    def uniform_sampling(self):
        if self.debug:
                print("uniform_sampling - Start")
                
        obs_1, act_1, obs_2, act_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size)
        
        if self.debug:
            print("uniform_sampling - After get_queries")
            print("obs_1 length:", len(obs_1))
            print("act_1 length:", len(act_1))
            print("obs_2 length:", len(obs_2))
            print("act_2 length:", len(act_2))
            print("r_t_1 shape:", r_t_1.shape)
            print("r_t_2 shape:", r_t_2.shape)

        # get labels
        obs_1, act_1, obs_2, act_2, r_t_1, r_t_2, labels= self.get_label(
            obs_1, act_1, r_t_1, obs_2, act_2, r_t_2)
        # if not self.vlm_label: 
        #     # get queries
        #     if not self.image_reward:
        #         obs_1, act_1, obs_2, act_2, r_t_1, r_t_2 =  self.get_queries(
        #             mb_size=self.mb_size)
        #         # get labels
        #         obs_1, act_1, r_t_1, obs_2, act_2, r_t_2, labels = self.get_label(
        #             obs_1, act_1, r_t_1, obs_2, act_2, r_t_2)
        #     else:
        #         obs_1, act_1, obs_2, act_2, r_t_1, r_t_2, img_t_1, img_t_2 =  self.get_queries(
        #             mb_size=self.mb_size)
        #         obs_1, act_1, r_t_1, obs_2, act_2, r_t_2, img_t_1, img_t_2, labels = self.get_label(
        #             obs_1, act_1, r_t_1, obs_2, act_2, r_t_2, img_t_1, img_t_2)
        # else:
        #     if self.cached_label_path is None:
        #         obs_1, act_1, obs_2, act_2, r_t_1, r_t_2, img_t_1, img_t_2 =  self.get_queries(
        #             mb_size=self.mb_size)
        #         if not self.image_reward:
        #             obs_1, act_1, r_t_1, obs_2, act_2, r_t_2, gt_labels, vlm_labels = self.get_label(
        #                 obs_1, act_1, r_t_1, obs_2, act_2, r_t_2, img_t_1, img_t_2)
        #         else:
        #             obs_1, act_1, r_t_1, obs_2, act_2, r_t_2, img_t_1, img_t_2, gt_labels, vlm_labels = self.get_label(
        #                 obs_1, act_1, r_t_1, obs_2, act_2, r_t_2, img_t_1, img_t_2)
        #     else:
        #         if self.read_cache_idx < len(self.all_cached_labels):
        #             combined_images_list, obs_1, act_1, obs_2, act_2, r_t_1, r_t_2, gt_labels, vlm_labels = self.get_label_from_cached_states()
        #             if self.image_reward:
        #                 num, height, width, _ = combined_images_list.shape
        #                 img_t_1 = combined_images_list[:, :, :width//2, :]
        #                 img_t_2 = combined_images_list[:, :, width//2:, :]
        #                 if 'Rope' not in self.env_name and \
        #                     'Water' not in self.env_name:
        #                     resized_img_t_1 = np.zeros((num, self.image_height, self.image_width, 3), dtype=np.uint8)
        #                     resized_img_t_2 = np.zeros((num, self.image_height, self.image_width, 3), dtype=np.uint8)
        #                     for idx in range(len(img_t_1)):
        #                         resized_img_t_1[idx] = cv2.resize(img_t_1[idx], (self.image_height, self.image_width))
        #                         resized_img_t_2[idx] = cv2.resize(img_t_2[idx], (self.image_height, self.image_width))
        #                     img_t_1 = resized_img_t_1
        #                     img_t_2 = resized_img_t_2
        #         else:
        #             vlm_labels = []
                
            # labels = vlm_labels
        if self.debug:
            print("uniform_sampling - After get_label")
            print("obs_1 length:", len(obs_1))
            print("act_1 length:", len(act_1))
            print("obs_2 length:", len(obs_2))
            print("act_2 length:", len(act_2))
            print("r_t_1 shape:", r_t_1.shape)
            print("r_t_2 shape:", r_t_2.shape)
            print("labels shape:", labels.shape) 
        if len(labels) > 0:
            self.put_queries(obs_1, act_1, obs_2, act_2, labels)
            if self.debug:
                print("uniform_sampling - After put_queries")
                print("buffer_index:", self.buffer_index)
                print("buffer_seg1 shape:", self.buffer_seg1[self.buffer_index-1].x.shape)
                print("buffer_seg2 shape:", self.buffer_seg2[self.buffer_index-1].x.shape)
                print("buffer_act1 shape:", len(self.buffer_act1[self.buffer_index-1]))
                print("buffer_act2 shape:", len(self.buffer_act2[self.buffer_index-1]))
                print("buffer_label shape:", self.buffer_label[self.buffer_index-1].shape)
            # if not self.image_reward:
            #     self.put_queries(obs_1, act_1, obs_2, act_2, labels)
            # else:
            #     self.put_queries(img_t_1[:, ::self.resize_factor, ::self.resize_factor, :], img_t_2[:, ::self.resize_factor, ::self.resize_factor, :], labels)

        return len(labels)
    
    def get_label_from_cached_states(self):
        if self.read_cache_idx >= len(self.all_cached_labels):
            return None, None, None, None, None, []
        with open(self.all_cached_labels[self.read_cache_idx], 'rb') as f:
            data = pkl.load(f)
        combined_images_list, rational_labels, vlm_labels, obs_1, act_1, obs_2, act_2, r_t_1, r_t_2 = data
        self.read_cache_idx += 1
        return combined_images_list, obs_1, act_1, obs_2, act_2, r_t_1, r_t_2, rational_labels, vlm_labels
    
    def train_reward(self):
        self.train_times += 1

        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                obs_1 = []
                obs_2 = []
                act_1 = []
                act_2 = []
                for i in range(len(idxs)):
                    obs_1.append(copy.deepcopy(self.buffer_seg1[idxs[i]]))
                    obs_2.append(copy.deepcopy(self.buffer_seg2[idxs[i]]))
                    act_1.append(copy.deepcopy(self.buffer_act1[idxs[i]]))
                    act_2.append(copy.deepcopy(self.buffer_act2[idxs[i]]))
                
                if self.debug:
                    print(idxs, "IDX")
                    print(f"train_reward - Member {member} Epoch {epoch} idxs:", idxs)
                    print(f"train_reward - Member {member} Epoch {epoch} obs_1 length:", len(obs_1))
                    print(f"train_reward - Member {member} Epoch {epoch} obs_2 length:", len(obs_2))
                    print(f"train_reward - Member {member} Epoch {epoch} act_1 length:", len(act_1))
                    print(f"train_reward - Member {member} Epoch {epoch} act_2 length:", len(act_2))
                obs_1, act_1, obs_2, act_2 = self.process(obs_1, act_1, obs_2, act_2)
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if self.debug:
                    print(f"train_reward - Member {member} Epoch {epoch} obs_1.x shape:", obs_1.x.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} obs_2.x shape:", obs_2.x.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} act_1 shape:", act_1.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} act_2 shape:", act_2.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} labels shape:", labels.shape)
                
                if member == 0:
                    total += labels.size(0)
                
                # if self.image_reward:
                #     # obs_1 is batch_size x segment x image_height x image_width x 3
                #     obs_1 = np.transpose(obs_1, (0, 1, 4, 2, 3)) # for torch we need to transpose channel first
                #     obs_2 = np.transpose(obs_2, (0, 1, 4, 2, 3)) 
                #     # also we stored uint8 images, we need to convert them to float32
                #     obs_1 = obs_1.astype(np.float32) / 255.0
                #     obs_2 = obs_2.astype(np.float32) / 255.0
                #     obs_1 = obs_1.squeeze(1)
                #     obs_2 = obs_2.squeeze(1)

                # get logits
                r_hat1 = self.r_hat_member(obs_1, act_1, member=member)
                r_hat2 = self.r_hat_member(obs_2, act_2, member=member)
                r_hat1 = r_hat1.sum(axis=1,keepdim=True)
                r_hat2 = r_hat2.sum(axis=1,keepdim=True)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                if self.args.use_wandb: wandb.log({f"reward_model_{member}_loss": curr_loss.item()})
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
            if epoch % 20 == 0:
                print(f"train - Epoch {epoch} Loss: {loss.item()}")
                test_score = self.eval_test_data()
                print(f"train - Epoch {epoch} Test Score: {test_score}")
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        torch.cuda.empty_cache()
        
        return ensemble_acc
    



class RewardModelImage:
    def __init__(self, ds, da, args,
                 ensemble_size=3, lr=3e-4, mb_size = 64, size_segment=1, 
                 max_size=5000, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0,
                 
                # vlm related params
                vlm_label=True,
                env_name="DressingEnv",
                vlm="gemini_free_form",
                clip_prompt=None,
                log_dir=None,
                flip_vlm_label=False,
                save_query_interval=25,
                cached_label_path=None,
                use_action = False,
                # image based reward
                reward_model_layers=3,
                reward_model_H=256,
                image_reward=True,
                image_height=128,
                image_width=128,
                resize_factor=1,
                resnet=False,
                conv_kernel_sizes=[5, 3, 3 ,3],
                conv_n_channels=[16, 32, 64, 128],
                conv_strides=[3, 2, 2, 2],

                **kwargs
                ):
        
        # train data is trajectories, must process to sa and s..   
        self.ds = ds #state dimension
        self.da = da#action dimension
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        self.args = args
        self.device = device
        print("Reward Model -- Device: ", self.device)
        self.gripper_idx = 2
        self.use_action = use_action
        self.encoder_type = args.encoder_type
        self.encoder_feature_dim = args.encoder_feature_dim
        self.gt_reward = False
        self.capacity = int(capacity)
        self.reward_model_layers = reward_model_layers
        self.reward_model_H = args.hidden_dim 

        
        self.buffer_seg1 = [_ for _ in range(self.capacity)]
        self.buffer_act1 = [_ for _ in range(self.capacity)]
        self.buffer_img1 = [_ for _ in range(self.capacity)]

        self.buffer_seg2 = [_ for _ in range(self.capacity)]
        self.buffer_act2 = [_ for _ in range(self.capacity)]
        self.buffer_img2 = [_ for _ in range(self.capacity)]
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
                
        self.construct_ensemble()
        self.inputs = []  # Store observations
        self.actions = []  # Store actions
        self.targets = []
        self.images = []
        
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 32
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch
        
        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin

        # vlm label
        self.vlm_label = vlm_label
        self.env_name = env_name
        self.vlm = vlm
        self.clip_prompt = clip_prompt
        self.vlm_label_acc = 0
        self.log_dir = args.work_dir
        self.flip_vlm_label = flip_vlm_label
        self.train_times = 0
        self.save_query_interval = save_query_interval
        
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        
        
        self.read_cache_idx = 0
        if cached_label_path is not None:
            self.cached_label_path = "{}/{}".format(dir_path, cached_label_path)
            all_cached_labels = sorted(os.listdir(self.cached_label_path))
            self.all_cached_labels = [os.path.join(self.cached_label_path, x) for x in all_cached_labels]
        else:
            self.cached_label_path = None
            
        self.debug = False
        
        print("Initialized Reward Model")
    def eval(self,):
        for i in range(self.de):
            self.ensemble[i].eval()

    def train(self,):
        for i in range(self.de):
            self.ensemble[i].train()
    
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
        
    def construct_ensemble(self):
        for i in range(self.de):
            model = PC2Reward(self.ds, self.da, use_action=self.use_action, hidden_dim=self.args.hidden_dim, encoder_type=self.args.encoder_type, encoder_feature_dim=self.encoder_feature_dim, num_layers=self.args.num_layers, num_filters=self.args.num_filters, args=self.args).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
            
    def add_data(self, obs, act, rew = None, img= None):

        self.inputs.append(obs)
        self.actions.append(act)
        if self.gt_reward:
            self.targets.append(rew)
        if img is not None:
            img = np.array(img)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images.append(image)
    
    def prepare_test_data(self,):
        for i in tqdm(range(10)):
            self.uniform_sampling()
        self.test_buffer_seg1 = self.buffer_seg1
        self.test_buffer_seg2 = self.buffer_seg2
        self.test_buffer_act1 = self.buffer_act1
        self.test_buffer_act2 = self.buffer_act2
        self.test_buffer_label = self.buffer_label
        self.test_buffer_index = self.buffer_index
        self.test_buffer_full = self.buffer_full
        self.buffer_seg1 = [_ for _ in range(self.capacity)]
        self.buffer_act1 = [_ for _ in range(self.capacity)]
        self.buffer_seg2 = [_ for _ in range(self.capacity)]
        self.buffer_act2 = [_ for _ in range(self.capacity)]
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
        print("test data prepared")
        
    def eval_test_data(self,):
        self.eval()
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.test_buffer_full else self.test_buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = 10 # int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                obs_1 = []
                obs_2 = []
                act_1 = []
                act_2 = []
                for i in range(len(idxs)):
                    obs_1.append(copy.deepcopy(self.test_buffer_seg1[idxs[i]]))
                    obs_2.append(copy.deepcopy(self.test_buffer_seg2[idxs[i]]))
                    act_1.append(copy.deepcopy(self.test_buffer_act1[idxs[i]]))
                    act_2.append(copy.deepcopy(self.test_buffer_act2[idxs[i]]))
                
                if self.debug:
                    print(idxs, "IDX")
                    print(f"train_reward - Member {member} Epoch {epoch} idxs:", idxs)
                    print(f"train_reward - Member {member} Epoch {epoch} obs_1 length:", len(obs_1))
                    print(f"train_reward - Member {member} Epoch {epoch} obs_2 length:", len(obs_2))
                    print(f"train_reward - Member {member} Epoch {epoch} act_1 length:", len(act_1))
                    print(f"train_reward - Member {member} Epoch {epoch} act_2 length:", len(act_2))
                obs_1, act_1, obs_2, act_2 = self.process(obs_1, act_1, obs_2, act_2)
                labels = self.test_buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if self.debug:
                    print(f"train_reward - Member {member} Epoch {epoch} obs_1.x shape:", obs_1.x.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} obs_2.x shape:", obs_2.x.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} act_1 shape:", act_1.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} act_2 shape:", act_2.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} labels shape:", labels.shape)
                
                if member == 0:
                    total += labels.size(0)
                
                # if self.image_reward:
                #     # obs_1 is batch_size x segment x image_height x image_width x 3
                #     obs_1 = np.transpose(obs_1, (0, 1, 4, 2, 3)) # for torch we need to transpose channel first
                #     obs_2 = np.transpose(obs_2, (0, 1, 4, 2, 3)) 
                #     # also we stored uint8 images, we need to convert them to float32
                #     obs_1 = obs_1.astype(np.float32) / 255.0
                #     obs_2 = obs_2.astype(np.float32) / 255.0
                #     obs_1 = obs_1.squeeze(1)
                #     obs_2 = obs_2.squeeze(1)

                # get logits
                r_hat1 = self.r_hat_member(obs_1, act_1, member=member)
                r_hat2 = self.r_hat_member(obs_2, act_2, member=member)
                r_hat1 = r_hat1.sum(axis=1,keepdim=True)
                r_hat2 = r_hat2.sum(axis=1,keepdim=True)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
        
        ensemble_acc = ensemble_acc / total
        if self.args.use_wandb:
            wandb.log({"test-accuracy": np.mean(ensemble_acc)})
        print("test-accuracy:", np.mean(ensemble_acc))
        self.train() 
        return ensemble_acc
          
    def add_data_batch(self, obses, actions, rewards = None, imges = None):
        for i in range(len(obses)):
            if self.gt_reward:
                self.add_data(obses[i], actions[i], rewards[i], None)
            else:
                self.add_data(obses[i], actions[i], None, imges[i])
    def process(self,obs_1, act_1, obs_2, act_2, device='cuda'):
        self.device = device
        if self.debug:
            print("process - Size of inputs as obs_1:", len(obs_1))
            print("process - Size of inputs as obs_2:", len(obs_2))
            print("process - Size of inputs as act_1:", len(act_1))
            print("process - Size of inputs as act_2:", len(act_2))
        # obs_1 = copy.deepcopy(obs_1)
        # obs_2 = copy.deepcopy(obs_2)
        # act_1 = copy.deepcopy(act_1)
        # act_2 = copy.deepcopy(act_2)
        if isinstance(obs_1[0], Data):
            # print("process - obs_1[0].x.shape:", obs_1[0].x.shape)
            obs_1 = Batch.from_data_list(obs_1)
            obs_2 = Batch.from_data_list(obs_2)
        else:
            raise ValueError("obs_1 and obs_2 must be lists of Data objects.")
    
        if 'flow' in self.args.encoder_type:
            if self.args.action_mode == 'translation' or (self.args.action_mode == 'rotation' and self.args.flow_q_mode in ['all_flow', 'repeat_gripper', 'concat_latent']):
                act_1 = np.concatenate(act_1)
                act_2 = np.concatenate(act_2)
                act_1 = torch.from_numpy(act_1).float().to(self.device)
                act_2 = torch.from_numpy(act_2).float().to(self.device)
        else:
                act_1 = torch.as_tensor(act_1, device=self.device)
                act_2 = torch.as_tensor(act_2, device=self.device)
        
        if self.debug:
            print("process - Size after making into batch class obs1.x:", obs_1.x.shape)
            print("process - Size after making into batch class obs2.x:", obs_2.x.shape)
            print("process - Size after making into concat act1:", act_1.shape)
            print("process - Size after making into concat act2:", act_2.shape)
            print("process - Batch size of obs1:", obs_1.batch_size)
            print("process - Batch size of obs2:", obs_2.batch_size)
        
        # if 'flow' in self.args.encoder_type:
        #     assert obs_1.batch_size == obs_2.batch_size
        #     for b_idx in range(obs_1.batch_size):
        #             act_1[(obs_1.batch == b_idx)] = act_1[(obs_1.batch == b_idx) & (obs_1.x[:, self.gripper_idx] == 1)]
        #             act_2[(obs_2.batch == b_idx)] = act_2[(obs_2.batch == b_idx) & (obs_2.x[:, self.gripper_idx] == 1)]
        # self.device = "cuda"
        return obs_1, act_1, obs_2, act_2
    
    def get_rank_probability(self, obs_1, act_1, obs_2, act_2):
        # get probability obs_1, act_1 > obs_2, act_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(obs_1, act_1, obs_2, act_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_entropy(self, obs_1, act_1, obs_2, act_2):
        # get entropy of probability obs_1, act_1 > obs_2, act_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(obs_1, act_1, obs_2, act_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, obs_1, act_1, obs_2, act_2, member=-1):
        # softmaxing to get the probabilities
        with torch.no_grad():
            r_hat1 = self.r_hat_member(obs_1, act_1, member=member)
            r_hat2 = self.r_hat_member(obs_2, act_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        # taking 0 index for probability obs_1, act_1 > obs_2, act_2
        return F.softmax(r_hat, dim=-1)[:,0]
    
    def p_hat_entropy(self, obs_1, act_1, obs_2, act_2, member=-1):
        # softmaxing to get the probabilities
        with torch.no_grad():
            r_hat1 = self.r_hat_member(obs_1, act_1, member=member)
            r_hat2 = self.r_hat_member(obs_2, act_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, obs, act, member=-1):
        # the network parameterizes r hat
        return self.ensemble[member](obs.to(device), act.to(device))

    def r_hat(self, obs, act):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(obs, act, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, obs, act):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(obs, act, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def load(self, model_dir, step):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.join(file_dir, model_dir)
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member), map_location=device)
            )
        print("Finished loading reward model ensemble")
    
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            obs_1 = self.buffer_seg1[epoch*batch_size:last_index]
            act_1 = self.buffer_act1[epoch*batch_size:last_index]
            obs_2 = self.buffer_seg2[epoch*batch_size:last_index]
            act_2 = self.buffer_act2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(obs_1, act_1, member=member)
                r_hat2 = self.r_hat_member(obs_2, act_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
    def get_queries(self, mb_size=20): 
        # We sample a batch of segments and actions from inputs, actions and targets which represent all the data we have
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        max_len = len(self.inputs)
        # get train traj
        train_obses = copy.deepcopy(self.inputs[:max_len])# train_obses = np.array(self.inputs[:max_len])
        train_acts = copy.deepcopy(self.actions[:max_len])# train_acts = np.array(self.actions[:max_len])
        train_targets = copy.deepcopy(self.targets[:max_len])# train_targets = np.array(self.targets[:max_len])
        train_images = copy.deepcopy(self.images[:max_len])# train_images = np.array(self.images[:max_len])
        
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        obs_2_t = []
        act_2_t = []

        img_2_t = []
        for ind in batch_index_2:
            obs_2_t.append(train_obses[ind]) # Batch x T x dim of obs
            act_2_t.append(train_acts[ind]) # Batch x T x dim of act
            img_2_t.append(train_images[ind]) # Batch x T x *img_dim
            
        
        # print(len(obs_2_t), len(act_2_t), len(r_t_2_t), " LEN of sample 2")
        # print(len(obs_2_t[0]))
        # for i in range(len(obs_2_t)):
        #     print(obs_2_t[i].x.shape, len(act_2_t[i]), r_t_2_t[i], "LEN of each sample")
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        obs_1_t = []
        act_1_t = []
        self.gt_label = []
        img_1_t = []
        for ind in batch_index_1:
            obs_1_t.append(train_obses[ind]) # Batch x T x dim of obs
            act_1_t.append(train_acts[ind]) # Batch x T x dim of act 
            img_1_t.append(train_images[ind]) # Batch x T x *img_dim
        
        for i in range(len(batch_index_1)):
            if  0 < batch_index_2[i] - batch_index_1[i]:
                self.gt_label.append(1)
            elif 0 < batch_index_1[i] - batch_index_2[i]:
                self.gt_label.append(0)
            else:
                self.gt_label.append(-1)
        # print(len(obs_1_t), len(act_1_t), len(r_t_1_t), " LEN of sample 1")
        # obs_1 = []
        # act_1 = []
        # r_t_1 = []
        # obs_2 = []
        # act_2 = []
        # r_t_2 = []
        # # Generate time index 
        # for ind in range((len(obs_1_t))):
        #     random_idx_2 = np.random.choice(len_traj-self.size_segment)
        #     obs_2.append(obs_2_t[random_idx_2:random_idx_2+self.size_segment]) # Batch x size_seg x dim of obs
        #     act_2.append(act_2_t[random_idx_2:random_idx_2+self.size_segment]) # Batch x size_seg x dim of act
        #     r_t_2.append(r_t_2_t[random_idx_2:random_idx_2+self.size_segment]) # Batch x size_seg x 1
        #     random_idx_1 = np.random.choice(len_traj-self.size_segment)
        #     obs_1.append(obs_1_t[random_idx_1:random_idx_1+self.size_segment]) # Batch x size_seg x dim of obs
        #     act_1.append(act_1_t[random_idx_1:random_idx_1+self.size_segment]) # Batch x size_seg x dim of act
        #     r_t_1.append(r_t_1_t[random_idx_1:random_idx_1+self.size_segment]) # Batch x size_seg x 1

        # obs_1 = np.array(obs_1)
        # print(act_1[0])
        # act_1 = np.array(act_1_t)
    
        # obs_2 = np.array(obs_2)
        # act_2 = np.array(act_2_t)
 
        if self.debug:
            print("get_queries - obs_1_t:", len(obs_1_t), "obs_2_t:", len(obs_2_t), "act_1_t:", len(act_1_t), "act_2_t:", len(act_2_t))
            print("get_queries - obs_1_t[0] shape:", obs_1_t[0].x.shape, "obs_2_t[0] shape:", obs_2_t[0].x.shape, "act_1_t[0] shape:", len(act_1_t[0]), "act_2_t[0] shape:", len(act_2_t[0]))

        return obs_1_t, act_1_t, obs_2_t, act_2_t, img_1_t, img_2_t
    
    def put_queries(self, obs_1, act_1, obs_2, act_2, labels):
        # put queries into the buffer that is bufferseg this is different from the entire dataset. The aim of this buffer is for backpropagation for the reward model
        total_sample = len(obs_1)
        next_index = self.buffer_index + total_sample
        if self.debug:
            print("put_queries - total_sample:", total_sample)
            print("put_queries - next_index:", next_index)
            print("put_queries - buffer_index:", self.buffer_index)
            print("put_queries - obs_1[0] shape:", obs_1[0].x.shape)
            print("put_queries - obs_2[0] shape:", obs_2[0].x.shape)
            print("put_queries - act_1 shape:", len(act_1[0]))
            print("put_queries - act_2 shape:", len(act_2[0]))
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            for i in range(self.buffer_index, self.capacity):
                self.buffer_seg1[i] = obs_1[i]
                self.buffer_seg2[i] = obs_2[i]
                self.buffer_act1[i] = act_1[i]
                self.buffer_act2[i] = act_2[i]
            # np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], obs_1[:maximum_index])
            # np.copyto(self.buffer_act1[self.buffer_index:self.capacity], act_1[:maximum_index])
            # np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], obs_2[:maximum_index])
            # np.copyto(self.buffer_act2[self.buffer_index:self.capacity], act_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                for i in range(remain):
                    self.buffer_seg1[i] = obs_1[maximum_index + i]
                    self.buffer_seg2[i] = obs_2[maximum_index + i]
                    self.buffer_act1[i] = act_1[maximum_index + i]
                    self.buffer_act2[i] = act_2[maximum_index + i]
                # np.copyto(self.buffer_seg1[0:remain], obs_1[maximum_index:])
                # np.copyto(self.buffer_act1[0:remain], act_1[maximum_index:])
                # np.copyto(self.buffer_seg2[0:remain], obs_2[maximum_index:])
                # np.copyto(self.buffer_act2[0:remain], act_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            for i in range(total_sample):
                self.buffer_seg1[self.buffer_index+i] = obs_1[i]
                self.buffer_seg2[self.buffer_index+i] = obs_2[i]
                # print(len(act_1), len(act_2), len(labels), "LEN of act1, act2, labels")
                # print(len(self.buffer_act1), len(self.buffer_act2), len(self.buffer_label), "LEN of buffer act1, act2, labels")
                self.buffer_act1[self.buffer_index+i] = act_1[i]
                self.buffer_act2[self.buffer_index+i] = act_2[i]
            # np.copyto(self.buffer_seg1[self.buffer_index:next_index], obs_1)
            # np.copyto(self.buffer_act1[self.buffer_index:next_index], act_1)
            # np.copyto(self.buffer_seg2[self.buffer_index:next_index], obs_2)
            # np.copyto(self.buffer_act2[self.buffer_index:next_index], act_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index
            
        if self.debug:
            print("put_queries - bufferseg1 shape:", self.buffer_seg1[self.buffer_index-1].x.shape)
            print("put_queries - bufferseg2 shape:", self.buffer_seg2[self.buffer_index-1].x.shape)
            print("put_queries - bufferact1 shape:", len(self.buffer_act1[self.buffer_index-1]))
            print("put_queries - bufferact2 shape:", len(self.buffer_act2[self.buffer_index-1]))
        
            
    def get_label(self, obs_1, act_1, img_t_1, obs_2, act_2, img_t_2):
        # get the label for the pair of obs_1, act_1 and obs_2, act_2
        
        # # skip the query
        # if self.teacher_thres_skip > 0: 
        #     max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
        #     max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
        #     if sum(max_index) == 0:
        #         return None, None, None, None, None, None, []

        #     obs_1 = obs_1[max_index]
        #     act_1 = act_1[max_index]
        #     r_t_1 = r_t_1[max_index]
        #     obs_2 = obs_2[max_index]
        #     act_2 = act_2[max_index]
        #     r_t_2 = r_t_2[max_index]
        #     sum_r_t_1 = np.sum(r_t_1, axis=1)
        #     sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # equally preferable
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

        gpt_two_image_paths = []
        combined_images_list = []
        useful_indices = []
        rational_labels = []
        
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        save_path = "{}/data/gpt_query_image/{}/{}".format(dir_path, self.env_name, time_string)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        for idx, (img1, img2) in enumerate(zip(img_t_1, img_t_2)):
            combined_image = np.concatenate([img1, img2], axis=1)
            combined_images_list.append(combined_image)
            combined_image = Image.fromarray(combined_image)
            first_image_save_path = os.path.join(save_path, "first_{:06}.png".format(idx))
            second_image_save_path = os.path.join(save_path, "second_{:06}.png".format(idx))
            Image.fromarray(img1).save(first_image_save_path)
            Image.fromarray(img2).save(second_image_save_path)
            gpt_two_image_paths.append([first_image_save_path, second_image_save_path])
            

            diff = np.linalg.norm(img1 - img2)
            if diff < 1e-3: # ignore the pair if the image is exactly the same
                useful_indices.append(0)
            else:
                useful_indices.append(1)
                    
        vlm_labels = []
        from dressing_motion.curl.vlm_reward.gpt4_infer import gpt4v_infer_2
        wandb.log({"Query Prompt": gpt_free_query_env_prompts[self.env_name]})
        for idx, (img_path_1, img_path_2) in enumerate(gpt_two_image_paths):
            print("querying vlm {}/{}".format(idx, len(gpt_two_image_paths)))
            query_prompt = gpt_free_query_env_prompts[self.env_name]
            summary_prompt = gpt_summary_env_prompts[self.env_name]
            res = gpt4v_infer_2(query_prompt, summary_prompt, img_path_1, img_path_2)
            try:
                label_res = int(res)
            except:
                label_res = -1
            print("index:", idx)
            print("ground truth label:", self.gt_label[idx])
            print("Getting VLM label:", label_res)
            vlm_labels.append(label_res)
            time.sleep(0.1)
        # for idx, (img1, img2) in enumerate(zip(img_t_1, img_t_2)):
        #     res = gemini_query_2(
        #             [
        #                 gemini_free_query_prompt1,
        #                 Image.fromarray(img1), 
        #                 gemini_free_query_prompt2,
        #                 Image.fromarray(img2), 
        #                 gemini_free_query_env_prompts[self.env_name]
        #     ],
        #                 gemini_summary_env_prompts[self.env_name]
        #     )
        #     try:
        #         res = int(res)
        #         print("index:", idx)
        #         print("ground truth label:", self.gt_label[idx])
        #         print("Getting VLM label:", res)
        #         if res not in [0, 1, -1]:
        #             res = -1
        #     except:
        #         res = -1
        #     vlm_labels.append(res)   
        count = 0
        tot = 0
        for i in range(len(vlm_labels)):
            if vlm_labels[i] == -1 or self.gt_label[i] == -1:
                continue
            if vlm_labels[i] == self.gt_label[i]:
                count += 1
            tot += 1
        print("vlm label accuracy:", count/tot)
        wandb.log({"vlm_label_accuracy": count/tot})
        exit()
        vlm_labels = np.array(vlm_labels).reshape(-1, 1)
        good_idx = (vlm_labels != -1).flatten()
        useful_indices = (np.array(useful_indices) == 1).flatten()
        good_idx = np.logical_and(good_idx, useful_indices)
        obs1 = []
        act1 = []
        obs2 = []
        act2 = []
        img1 = []
        img2 = []
        labels = []
        combined_images_list1 = []
        # print("vlm_labels:", vlm_labels)
        # print("good_idx:", good_idx)
        for ind in range(len(good_idx)):
            if good_idx[ind]:
                obs1.append(obs_1[ind])
                act1.append(act_1[ind])
                obs2.append(obs_2[ind])
                act2.append(act_2[ind])
                img1.append(img_t_1[ind])
                img2.append(img_t_2[ind])
                labels.append(vlm_labels[ind])
                print("vlm_label:", vlm_labels[ind])
                combined_images_list1.append(combined_images_list[ind])
        # print(labels)
        if self.flip_vlm_label:
            labels = 1 - labels
            

        if self.train_times % self.save_query_interval == 0 or 'gpt4v' in self.vlm:
            save_path = os.path.join(self.log_dir, "vlm_label_set")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open("{}/{}.pkl".format(save_path, time_string), "wb") as f:
                pkl.dump([combined_images_list1, labels, obs1, act1, obs2, act2, img1, img2], f, protocol=pkl.HIGHEST_PROTOCOL)

        if len(labels) > 0:
            print("vlm label obtained")
            print("vlm label:", labels)
        else:
            print("no vlm label")
        labels = np.array(labels).reshape(-1, 1)
        # print("labels shape:", labels.shape)
        # print("obs1 shape:", len(obs1))
        return obs1, act1, obs2, act2, img1, img2,  labels
        

        # # if not self.image_reward:
        # labels =  np.array(labels).reshape(-1, 1)
        # # print(labels.shape, "LABEL SHAPE")
        # return obs_1, act_1, obs_2, act_2, r_t_1, r_t_2, labels
        # # else:
        # #     return obs_1, act_1, obs_2, act_2, r_t_1, r_t_2, img_t_1, img_t_2, labels
 
    
    def uniform_sampling(self):
        if self.debug:
                print("uniform_sampling - Start")
                
        obs_1, act_1, obs_2, act_2, img_1_t, img_2_t =  self.get_queries(
            mb_size=self.mb_size)
        
        if self.debug:
            print("uniform_sampling - After get_queries")
            print("obs_1 length:", len(obs_1))
            print("act_1 length:", len(act_1))
            print("obs_2 length:", len(obs_2))
            print("act_2 length:", len(act_2))


        # get labels
        obs_1, act_1, obs_2, act_2, img_1_t, img_2_t, labels= self.get_label(
            obs_1, act_1, img_1_t, obs_2, act_2, img_2_t)
        # if not self.vlm_label: 
        #     # get queries
        #     if not self.image_reward:
        #         obs_1, act_1, obs_2, act_2, r_t_1, r_t_2 =  self.get_queries(
        #             mb_size=self.mb_size)
        #         # get labels
        #         obs_1, act_1, r_t_1, obs_2, act_2, r_t_2, labels = self.get_label(
        #             obs_1, act_1, r_t_1, obs_2, act_2, r_t_2)
        #     else:
        #         obs_1, act_1, obs_2, act_2, r_t_1, r_t_2, img_t_1, img_t_2 =  self.get_queries(
        #             mb_size=self.mb_size)
        #         obs_1, act_1, r_t_1, obs_2, act_2, r_t_2, img_t_1, img_t_2, labels = self.get_label(
        #             obs_1, act_1, r_t_1, obs_2, act_2, r_t_2, img_t_1, img_t_2)
        # else:
        #     if self.cached_label_path is None:
        #         obs_1, act_1, obs_2, act_2, r_t_1, r_t_2, img_t_1, img_t_2 =  self.get_queries(
        #             mb_size=self.mb_size)
        #         if not self.image_reward:
        #             obs_1, act_1, r_t_1, obs_2, act_2, r_t_2, gt_labels, vlm_labels = self.get_label(
        #                 obs_1, act_1, r_t_1, obs_2, act_2, r_t_2, img_t_1, img_t_2)
        #         else:
        #             obs_1, act_1, r_t_1, obs_2, act_2, r_t_2, img_t_1, img_t_2, gt_labels, vlm_labels = self.get_label(
        #                 obs_1, act_1, r_t_1, obs_2, act_2, r_t_2, img_t_1, img_t_2)
        #     else:
        #         if self.read_cache_idx < len(self.all_cached_labels):
        #             combined_images_list, obs_1, act_1, obs_2, act_2, r_t_1, r_t_2, gt_labels, vlm_labels = self.get_label_from_cached_states()
        #             if self.image_reward:
        #                 num, height, width, _ = combined_images_list.shape
        #                 img_t_1 = combined_images_list[:, :, :width//2, :]
        #                 img_t_2 = combined_images_list[:, :, width//2:, :]
        #                 if 'Rope' not in self.env_name and \
        #                     'Water' not in self.env_name:
        #                     resized_img_t_1 = np.zeros((num, self.image_height, self.image_width, 3), dtype=np.uint8)
        #                     resized_img_t_2 = np.zeros((num, self.image_height, self.image_width, 3), dtype=np.uint8)
        #                     for idx in range(len(img_t_1)):
        #                         resized_img_t_1[idx] = cv2.resize(img_t_1[idx], (self.image_height, self.image_width))
        #                         resized_img_t_2[idx] = cv2.resize(img_t_2[idx], (self.image_height, self.image_width))
        #                     img_t_1 = resized_img_t_1
        #                     img_t_2 = resized_img_t_2
        #         else:
        #             vlm_labels = []
                
            # labels = vlm_labels
        if self.debug:
            print("uniform_sampling - After get_label")
            print("obs_1 length:", len(obs_1))
            print("act_1 length:", len(act_1))
            print("obs_2 length:", len(obs_2))
            print("act_2 length:", len(act_2))
  
            print("labels shape:", labels.shape) 
        if len(labels) > 0:
            self.put_queries(obs_1, act_1, obs_2, act_2, labels)
            if self.debug:
                print("uniform_sampling - After put_queries")
                print("buffer_index:", self.buffer_index)
                print("buffer_seg1 shape:", self.buffer_seg1[self.buffer_index-1].x.shape)
                print("buffer_seg2 shape:", self.buffer_seg2[self.buffer_index-1].x.shape)
                print("buffer_act1 shape:", len(self.buffer_act1[self.buffer_index-1]))
                print("buffer_act2 shape:", len(self.buffer_act2[self.buffer_index-1]))
                print("buffer_label shape:", self.buffer_label[self.buffer_index-1].shape)
            # if not self.image_reward:
            #     self.put_queries(obs_1, act_1, obs_2, act_2, labels)
            # else:
            #     self.put_queries(img_t_1[:, ::self.resize_factor, ::self.resize_factor, :], img_t_2[:, ::self.resize_factor, ::self.resize_factor, :], labels)

        return len(labels)
    
    def get_label_from_cached_states(self):
        if self.read_cache_idx >= len(self.all_cached_labels):
            return None, None, None, None, None, []
        with open(self.all_cached_labels[self.read_cache_idx], 'rb') as f:
            data = pkl.load(f)
        combined_images_list, labels, obs_1, act_1, obs_2, act_2, img_t_1, img_t_2 = data
        self.read_cache_idx += 1
        return combined_images_list, obs_1, act_1, obs_2, act_2, img_t_1, img_t_2, labels
    
    def train_reward(self):
        self.train_times += 1

        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                obs_1 = []
                obs_2 = []
                act_1 = []
                act_2 = []
                for i in range(len(idxs)):
                    obs_1.append(copy.deepcopy(self.buffer_seg1[idxs[i]]))
                    obs_2.append(copy.deepcopy(self.buffer_seg2[idxs[i]]))
                    act_1.append(copy.deepcopy(self.buffer_act1[idxs[i]]))
                    act_2.append(copy.deepcopy(self.buffer_act2[idxs[i]]))
                
                if self.debug:
                    print(idxs, "IDX")
                    print(f"train_reward - Member {member} Epoch {epoch} idxs:", idxs)
                    print(f"train_reward - Member {member} Epoch {epoch} obs_1 length:", len(obs_1))
                    print(f"train_reward - Member {member} Epoch {epoch} obs_2 length:", len(obs_2))
                    print(f"train_reward - Member {member} Epoch {epoch} act_1 length:", len(act_1))
                    print(f"train_reward - Member {member} Epoch {epoch} act_2 length:", len(act_2))
                obs_1, act_1, obs_2, act_2 = self.process(obs_1, act_1, obs_2, act_2)
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if self.debug:
                    print(f"train_reward - Member {member} Epoch {epoch} obs_1.x shape:", obs_1.x.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} obs_2.x shape:", obs_2.x.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} act_1 shape:", act_1.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} act_2 shape:", act_2.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} labels shape:", labels.shape)
                
                if member == 0:
                    total += labels.size(0)
                
                # if self.image_reward:
                #     # obs_1 is batch_size x segment x image_height x image_width x 3
                #     obs_1 = np.transpose(obs_1, (0, 1, 4, 2, 3)) # for torch we need to transpose channel first
                #     obs_2 = np.transpose(obs_2, (0, 1, 4, 2, 3)) 
                #     # also we stored uint8 images, we need to convert them to float32
                #     obs_1 = obs_1.astype(np.float32) / 255.0
                #     obs_2 = obs_2.astype(np.float32) / 255.0
                #     obs_1 = obs_1.squeeze(1)
                #     obs_2 = obs_2.squeeze(1)

                # get logits
                r_hat1 = self.r_hat_member(obs_1, act_1, member=member)
                r_hat2 = self.r_hat_member(obs_2, act_2, member=member)
                r_hat1 = r_hat1.sum(axis=1,keepdim=True)
                r_hat2 = r_hat2.sum(axis=1,keepdim=True)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                if self.args.use_wandb: wandb.log({f"reward_model_{member}_loss": curr_loss.item()})
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
            # if epoch % 20 == 0 and epoch > 0:
            #     print(f"train - Epoch {epoch} Loss: {loss.item()}")
            #     test_score = self.eval_test_data()
            #     print(f"train - Epoch {epoch} Test Score: {test_score}")
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        torch.cuda.empty_cache()
        
        return ensemble_acc




class RewardModelVLM:
    def __init__(self, ds, da, args,
                 ensemble_size=5, lr=3e-7, mb_size = 64, size_segment=1, 
                 max_size=5000, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0,
                 
                # vlm related params
                vlm_label=True,
                env_name="DressingEnv",
                vlm="gpt4v_two_image",
                clip_prompt=None,
                log_dir=None,
                flip_vlm_label=False,
                save_query_interval=25,
                cached_label_path=None,
                use_action = False,
                # image based reward
                reward_model_layers=3,
                reward_model_H=256,
                image_reward=True,
                image_height=128,
                image_width=128,
                resize_factor=1,
                resnet=False,
                conv_kernel_sizes=[5, 3, 3 ,3],
                conv_n_channels=[16, 32, 64, 128],
                conv_strides=[3, 2, 2, 2],

                **kwargs
                ):
        
        # train data is trajectories, must process to sa and s..   
        self.ds = ds #state dimension
        self.da = da#action dimension
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        self.args = args
        self.device = device
        print("Reward Model -- Device: ", self.device)
        self.gripper_idx = 2
        self.use_action = use_action
        self.encoder_type = args.encoder_type
        self.encoder_feature_dim = args.encoder_feature_dim
        self.gt_reward = False
        self.capacity = int(capacity)
        self.reward_model_layers = reward_model_layers
        self.reward_model_H = args.hidden_dim 

        
        self.buffer_seg1 = [_ for _ in range(self.capacity)]
        self.buffer_act1 = [_ for _ in range(self.capacity)]
        self.buffer_img1 = [_ for _ in range(self.capacity)]

        self.buffer_seg2 = [_ for _ in range(self.capacity)]
        self.buffer_act2 = [_ for _ in range(self.capacity)]
        self.buffer_img2 = [_ for _ in range(self.capacity)]
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
                
        self.construct_ensemble()
        self.inputs = []  # Store observations
        self.actions = []  # Store actions
        self.targets = []
        self.images = []
        
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 32
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch
        
        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin

        # vlm label
        self.vlm_label = vlm_label
        self.env_name = env_name
        self.vlm = vlm
        self.clip_prompt = clip_prompt
        self.vlm_label_acc = 0
        self.log_dir = args.work_dir
        self.flip_vlm_label = flip_vlm_label
        self.train_times = 0
        self.save_query_interval = save_query_interval
        
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        
        
        self.read_cache_idx = 0
        if cached_label_path is not None:
            self.cached_label_path = "{}/{}".format(dir_path, cached_label_path)
            all_cached_labels = sorted(os.listdir(self.cached_label_path))
            self.all_cached_labels = [os.path.join(self.cached_label_path, x) for x in all_cached_labels]
        else:
            self.cached_label_path = None
            
        self.debug = False
        
        print("Initialized Reward Model")
    def eval(self,):
        for i in range(self.de):
            self.ensemble[i].eval()

    def train(self,):
        for i in range(self.de):
            self.ensemble[i].train()
    
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
        
    def construct_ensemble(self):
        for i in range(self.de):
            model = PC2Reward(self.ds, self.da, use_action=self.use_action, hidden_dim=self.args.hidden_dim, encoder_type=self.args.encoder_type, encoder_feature_dim=self.encoder_feature_dim, num_layers=self.args.num_layers, num_filters=self.args.num_filters, args=self.args).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
    
    def add_data_from_cache(self, data_dir, fill_size = 12000):
        file_list = os.listdir(data_dir)
        file_list = sorted(file_list)
        file_list = [os.path.join(data_dir, x) for x in file_list]
        file_list = file_list[:len(file_list)//3]
        print(file_list)
        
        dict_1_list = []
        dict_2_list = []
        label_list = []
        gt_list = []
        for file in file_list:
            print("file:", file)
            with open(file, 'rb') as f:
                combined_images_list1, labels, dict_list_1, dict_list_2, gt_labels_1 = pkl.load(f)
            print("type of labels:", type(labels))
            print("Length of dict_1_list:", len(dict_list_1),"Length of dict_2_list:", len(dict_list_2), "Length of label_list:", len(labels))
            if len(dict_list_1) != len(dict_list_2) or len(dict_list_1) != len(labels):
                print("______APPROX__________")
                min_len = min(len(dict_list_1), len(dict_list_2), len(labels))
                dict_list_1 = dict_list_1[:min_len]
                dict_list_2 = dict_list_2[:min_len]
                labels = labels[:min_len]
                print("Length of dict_1_list:", len(dict_list_1),"Length of dict_2_list:", len(dict_list_2), "Length of label_list:", len(labels))
                print("Shape of labels:", labels.shape)
                if labels.shape[1] >= 1:
                    labels.reshape(-1, 1)
            labels = labels.reshape(-1)
            lab_list = labels.tolist()
            dict_1_list += dict_list_1
            dict_2_list += dict_list_2
            label_list +=  lab_list
            
        obs_1_list = []
        obs_2_list = []
        act_1_list = []
        act_2_list = []
        labels = []        
        batch_index = np.random.choice(len(dict_1_list), fill_size, replace=True)
        for idx in batch_index:
            dict_1 = dict_1_list[idx]
            dict_2 = dict_2_list[idx]
            label = copy.deepcopy(label_list[idx])
            len_1 = len(dict_1['obs'])
            len_2 = len(dict_2['obs'])
            
            ind_1 = np.random.choice(len_1, 1)[0]
            ind_2 = np.random.choice(len_2, 1)[0]
            
            # obs_1 =  copy.deepcopy(dict_1["obs"][ind_1])
            # obs_2 =  copy.deepcopy(dict_2["obs"][ind_2])
            
            # act_1 = copy.deepcopy(dict_1["action"][ind_1])
            # act_2 = copy.deepcopy(dict_2["action"][ind_2])

            obs_1 =  copy.deepcopy(dict_1["obs"])
            obs_1.x = torch.cat((obs_1.x, torch.zeros(obs_1.x.size(0), 1)), dim=1)
            obs_1.x[-1, 2] = copy.deepcopy(dict_1['total_force'])
            
            obs_2 =  copy.deepcopy(dict_2["obs"])
            obs_2.x = torch.cat((obs_2.x, torch.zeros(obs_2.x.size(0), 1)), dim=1)
            obs_2.x[-1, 2] = copy.deepcopy(dict_2['total_force'])

            
            act_1 = copy.deepcopy(dict_1["action"])
            act_2 = copy.deepcopy(dict_2["action"])

            obs_1_list.append(obs_1)
            obs_2_list.append(obs_2)
            act_1_list.append(act_1)
            act_2_list.append(act_2)
            labels.append(label)
        
        labels = np.array(labels).reshape(-1, 1)
        
        print("Data Loaded from Cache")
        self.put_queries(obs_1_list, act_1_list, obs_2_list, act_2_list, labels)        
        print("Data Added to Buffer")
        print("Testing training")
        self.train_reward()
    
    def put_queries(self, obs_1, act_1, obs_2, act_2, labels):
        # put queries into the buffer that is bufferseg this is different from the entire dataset. The aim of this buffer is for backpropagation for the reward model
        total_sample = len(obs_1)
        next_index = self.buffer_index + total_sample
        if self.debug:
            print("put_queries - total_sample:", total_sample)
            print("put_queries - next_index:", next_index)
            print("put_queries - buffer_index:", self.buffer_index)
            print("put_queries - obs_1[0] shape:", obs_1[0].x.shape)
            print("put_queries - obs_2[0] shape:", obs_2[0].x.shape)
            print("put_queries - act_1 shape:", len(act_1[0]))
            print("put_queries - act_2 shape:", len(act_2[0]))
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            for i in range(self.buffer_index, self.capacity):
                self.buffer_seg1[i] = obs_1[i]
                self.buffer_seg2[i] = obs_2[i]
                self.buffer_act1[i] = act_1[i]
                self.buffer_act2[i] = act_2[i]
            # np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], obs_1[:maximum_index])
            # np.copyto(self.buffer_act1[self.buffer_index:self.capacity], act_1[:maximum_index])
            # np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], obs_2[:maximum_index])
            # np.copyto(self.buffer_act2[self.buffer_index:self.capacity], act_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                for i in range(remain):
                    self.buffer_seg1[i] = obs_1[maximum_index + i]
                    self.buffer_seg2[i] = obs_2[maximum_index + i]
                    self.buffer_act1[i] = act_1[maximum_index + i]
                    self.buffer_act2[i] = act_2[maximum_index + i]
                # np.copyto(self.buffer_seg1[0:remain], obs_1[maximum_index:])
                # np.copyto(self.buffer_act1[0:remain], act_1[maximum_index:])
                # np.copyto(self.buffer_seg2[0:remain], obs_2[maximum_index:])
                # np.copyto(self.buffer_act2[0:remain], act_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            for i in range(total_sample):
                self.buffer_seg1[self.buffer_index+i] = obs_1[i]
                self.buffer_seg2[self.buffer_index+i] = obs_2[i]
                # print(len(act_1), len(act_2), len(labels), "LEN of act1, act2, labels")
                # print(len(self.buffer_act1), len(self.buffer_act2), len(self.buffer_label), "LEN of buffer act1, act2, labels")
                self.buffer_act1[self.buffer_index+i] = act_1[i]
                self.buffer_act2[self.buffer_index+i] = act_2[i]
            # np.copyto(self.buffer_seg1[self.buffer_index:next_index], obs_1)
            # np.copyto(self.buffer_act1[self.buffer_index:next_index], act_1)
            # np.copyto(self.buffer_seg2[self.buffer_index:next_index], obs_2)
            # np.copyto(self.buffer_act2[self.buffer_index:next_index], act_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index
            
        if self.debug:
            print("put_queries - bufferseg1 shape:", self.buffer_seg1[self.buffer_index-1].x.shape)
            print("put_queries - bufferseg2 shape:", self.buffer_seg2[self.buffer_index-1].x.shape)
            print("put_queries - bufferact1 shape:", len(self.buffer_act1[self.buffer_index-1]))
            print("put_queries - bufferact2 shape:", len(self.buffer_act2[self.buffer_index-1]))
            
        
    def prepare_test_data(self,):
        raise NotImplementedError
        
    def eval_test_data(self,):
        raise NotImplementedError
        self.eval()
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.test_buffer_full else self.test_buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = 10 # int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                obs_1 = []
                obs_2 = []
                act_1 = []
                act_2 = []
                for i in range(len(idxs)):
                    obs_1.append(copy.deepcopy(self.test_buffer_seg1[idxs[i]]))
                    obs_2.append(copy.deepcopy(self.test_buffer_seg2[idxs[i]]))
                    act_1.append(copy.deepcopy(self.test_buffer_act1[idxs[i]]))
                    act_2.append(copy.deepcopy(self.test_buffer_act2[idxs[i]]))
                
                if self.debug:
                    print(idxs, "IDX")
                    print(f"train_reward - Member {member} Epoch {epoch} idxs:", idxs)
                    print(f"train_reward - Member {member} Epoch {epoch} obs_1 length:", len(obs_1))
                    print(f"train_reward - Member {member} Epoch {epoch} obs_2 length:", len(obs_2))
                    print(f"train_reward - Member {member} Epoch {epoch} act_1 length:", len(act_1))
                    print(f"train_reward - Member {member} Epoch {epoch} act_2 length:", len(act_2))
                obs_1, act_1, obs_2, act_2 = self.process(obs_1, act_1, obs_2, act_2)
                labels = self.test_buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if self.debug:
                    print(f"train_reward - Member {member} Epoch {epoch} obs_1.x shape:", obs_1.x.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} obs_2.x shape:", obs_2.x.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} act_1 shape:", act_1.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} act_2 shape:", act_2.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} labels shape:", labels.shape)
                
                if member == 0:
                    total += labels.size(0)
                
                # if self.image_reward:
                #     # obs_1 is batch_size x segment x image_height x image_width x 3
                #     obs_1 = np.transpose(obs_1, (0, 1, 4, 2, 3)) # for torch we need to transpose channel first
                #     obs_2 = np.transpose(obs_2, (0, 1, 4, 2, 3)) 
                #     # also we stored uint8 images, we need to convert them to float32
                #     obs_1 = obs_1.astype(np.float32) / 255.0
                #     obs_2 = obs_2.astype(np.float32) / 255.0
                #     obs_1 = obs_1.squeeze(1)
                #     obs_2 = obs_2.squeeze(1)

                # get logits
                r_hat1 = self.r_hat_member(obs_1, act_1, member=member)
                r_hat2 = self.r_hat_member(obs_2, act_2, member=member)
                r_hat1 = r_hat1.sum(axis=1,keepdim=True)
                r_hat2 = r_hat2.sum(axis=1,keepdim=True)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
        
        ensemble_acc = ensemble_acc / total
        if self.args.use_wandb:
            wandb.log({"test-accuracy": np.mean(ensemble_acc)})
        print("test-accuracy:", np.mean(ensemble_acc))
        self.train() 
        return ensemble_acc

                
    def process(self,obs_1, act_1, obs_2, act_2, device='cuda'):
        self.device = device
        if self.debug:
            print("process - Size of inputs as obs_1:", len(obs_1))
            print("process - Size of inputs as obs_2:", len(obs_2))
            print("process - Size of inputs as act_1:", len(act_1))
            print("process - Size of inputs as act_2:", len(act_2))
        # obs_1 = copy.deepcopy(obs_1)
        # obs_2 = copy.deepcopy(obs_2)
        # act_1 = copy.deepcopy(act_1)
        # act_2 = copy.deepcopy(act_2)
        if isinstance(obs_1[0], Data):
            obs_1 = Batch.from_data_list(obs_1)
            obs_2 = Batch.from_data_list(obs_2)
        else:
            raise ValueError("obs_1 and obs_2 must be lists of Data objects.")
    
        if 'flow' in self.args.encoder_type:
            if self.args.action_mode == 'translation' or (self.args.action_mode == 'rotation' and self.args.flow_q_mode in ['all_flow', 'repeat_gripper', 'concat_latent']):
                act_1 = np.concatenate(act_1)
                act_2 = np.concatenate(act_2)
                act_1 = torch.from_numpy(act_1).float().to(self.device)
                act_2 = torch.from_numpy(act_2).float().to(self.device)
        else:
                act_1 = torch.as_tensor(act_1, device=self.device)
                act_2 = torch.as_tensor(act_2, device=self.device)
        
        if self.debug:
            print("process - Size after making into batch class obs1.x:", obs_1.x.shape)
            print("process - Size after making into batch class obs2.x:", obs_2.x.shape)
            print("process - Size after making into concat act1:", act_1.shape)
            print("process - Size after making into concat act2:", act_2.shape)
            print("process - Batch size of obs1:", obs_1.batch)
            print("process - Batch size of obs2:", obs_2.batch)
        
        # if 'flow' in self.args.encoder_type:
        #     assert obs_1.batch_size == obs_2.batch_size
        #     for b_idx in range(obs_1.batch_size):
        #             act_1[(obs_1.batch == b_idx)] = act_1[(obs_1.batch == b_idx) & (obs_1.x[:, self.gripper_idx] == 1)]
        #             act_2[(obs_2.batch == b_idx)] = act_2[(obs_2.batch == b_idx) & (obs_2.x[:, self.gripper_idx] == 1)]
        # self.device = "cuda"
        return obs_1, act_1, obs_2, act_2
    
    def get_rank_probability(self, obs_1, act_1, obs_2, act_2):
        # get probability obs_1, act_1 > obs_2, act_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(obs_1, act_1, obs_2, act_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_entropy(self, obs_1, act_1, obs_2, act_2):
        # get entropy of probability obs_1, act_1 > obs_2, act_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(obs_1, act_1, obs_2, act_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, obs_1, act_1, obs_2, act_2, member=-1):
        # softmaxing to get the probabilities
        with torch.no_grad():
            r_hat1 = self.r_hat_member(obs_1, act_1, member=member)
            r_hat2 = self.r_hat_member(obs_2, act_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        # taking 0 index for probability obs_1, act_1 > obs_2, act_2
        return F.softmax(r_hat, dim=-1)[:,0]
    
    def p_hat_entropy(self, obs_1, act_1, obs_2, act_2, member=-1):
        # softmaxing to get the probabilities
        with torch.no_grad():
            r_hat1 = self.r_hat_member(obs_1, act_1, member=member)
            r_hat2 = self.r_hat_member(obs_2, act_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, obs, act, member=-1):
        # the network parameterizes r hat
        return self.ensemble[member](obs.to(device), act.to(device))

    def r_hat(self, obs, act):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(obs, act, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, obs, act):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(obs, act, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step, acc):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s_%s.pt' % (model_dir, step, member, acc)
            )
            
    def load(self, model_dir, step):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.join(file_dir, model_dir)
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member), map_location=device)
            )
        print("Finished loading reward model ensemble")
    
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            obs_1 = self.buffer_seg1[epoch*batch_size:last_index]
            act_1 = self.buffer_act1[epoch*batch_size:last_index]
            obs_2 = self.buffer_seg2[epoch*batch_size:last_index]
            act_2 = self.buffer_act2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(obs_1, act_1, member=member)
                r_hat2 = self.r_hat_member(obs_2, act_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
    
    def train_reward(self):
        self.train_times += 1

        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                obs_1 = []
                obs_2 = []
                act_1 = []
                act_2 = []
                for i in range(len(idxs)):
                    obs_1.append(copy.deepcopy(self.buffer_seg1[idxs[i]]))
                    obs_2.append(copy.deepcopy(self.buffer_seg2[idxs[i]]))
                    act_1.append(copy.deepcopy(self.buffer_act1[idxs[i]]))
                    act_2.append(copy.deepcopy(self.buffer_act2[idxs[i]]))
                
                if self.debug:
                    print(idxs, "IDX")
                    print(f"train_reward - Member {member} Epoch {epoch} idxs:", idxs)
                    print(f"train_reward - Member {member} Epoch {epoch} obs_1 length:", len(obs_1))
                    print(f"train_reward - Member {member} Epoch {epoch} obs_2 length:", len(obs_2))
                    print(f"train_reward - Member {member} Epoch {epoch} act_1 length:", len(act_1))
                    print(f"train_reward - Member {member} Epoch {epoch} act_2 length:", len(act_2))
                obs_1, act_1, obs_2, act_2 = self.process(obs_1, act_1, obs_2, act_2)
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if self.debug:
                    print(f"train_reward - Member {member} Epoch {epoch} obs_1.x shape:", obs_1.x.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} obs_2.x shape:", obs_2.x.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} act_1 shape:", act_1.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} act_2 shape:", act_2.shape)
                    print(f"train_reward - Member {member} Epoch {epoch} labels shape:", labels.shape)
                
                if member == 0:
                    total += labels.size(0)
                
                # if self.image_reward:
                #     # obs_1 is batch_size x segment x image_height x image_width x 3
                #     obs_1 = np.transpose(obs_1, (0, 1, 4, 2, 3)) # for torch we need to transpose channel first
                #     obs_2 = np.transpose(obs_2, (0, 1, 4, 2, 3)) 
                #     # also we stored uint8 images, we need to convert them to float32
                #     obs_1 = obs_1.astype(np.float32) / 255.0
                #     obs_2 = obs_2.astype(np.float32) / 255.0
                #     obs_1 = obs_1.squeeze(1)
                #     obs_2 = obs_2.squeeze(1)

                # get logits
                r_hat1 = self.r_hat_member(obs_1, act_1, member=member)
                r_hat2 = self.r_hat_member(obs_2, act_2, member=member)
                r_hat1 = r_hat1.sum(axis=1,keepdim=True)
                r_hat2 = r_hat2.sum(axis=1,keepdim=True)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                if self.args.use_wandb: wandb.log({f"reward_model_{member}_loss": curr_loss.item()})
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
            # if epoch % 20 == 0 and epoch > 0:
            #     print(f"train - Epoch {epoch} Loss: {loss.item()}")
            #     test_score = self.eval_test_data()
            #     print(f"train - Epoch {epoch} Test Score: {test_score}")
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        torch.cuda.empty_cache()
        
        return ensemble_acc
    