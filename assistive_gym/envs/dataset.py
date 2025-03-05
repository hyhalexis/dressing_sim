import time
import numpy as np
import torch
import os

import torch_geometric
from torch_geometric.data import Data, Dataset, Batch
import os.path as osp
from utils import voxelize_pc
import scipy
from garment_idx_utils import *
import matplotlib.pyplot as plt
import pickle5 as pickle
import random

def compute_return_array(discount_factors, reward_array, n_step=1):
    discount_to_n_step_return = {}
    discount_to_future_return = {}
    T = len(reward_array)
    for discount in discount_factors:
        future_returns = np.zeros_like(reward_array)
        R = 0
        for t in reversed(range(T)):
            R = R * discount + reward_array[t]
            future_returns[t] = R
        discount_to_future_return[discount] = future_returns

        n_step_returns = np.zeros(T - n_step + 1)
        discounts = discount ** np.arange(n_step)
        for t in range(T - n_step + 1):
            n_step_returns[t] = np.dot(discounts, reward_array[t:t+n_step])
        discount_to_n_step_return[discount] = n_step_returns
        
    return discount_to_n_step_return, discount_to_future_return

def list_files(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def convert_sim_to_real(sim_points):
    sim_points = sim_points.reshape(-1, 3)
    real_points = np.zeros_like(sim_points)
    real_points[:, 0] = sim_points[:, 2]
    real_points[:, 1] = sim_points[:, 0]
    real_points[:, 2] = sim_points[:, 1]
    return real_points

def create_forces_history(arr, window_size=5):
    arr = np.asarray(arr)
    result = [arr[max(0, i - window_size + 1): i + 1] for i in range(len(arr))]
    
    result = [np.pad(subarr, (window_size - len(subarr), 0), mode='constant', constant_values=0) 
              if len(subarr) < window_size else subarr 
              for subarr in result]

    return np.array(result)

def load_data_by_name(self, path, args):
    obses = []
    next_obses = []
    actions = []
    rewards = []
    not_dones = []
    non_randomized_obses = []
    forces = []
    ori_obses = []

    with open(file, 'rb') as f:
        traj = pickle.load(f)

    for i in range(len(traj)):
        # Add transition components directly to lists
        transition = traj[i]
        next_transition = traj[i+1] if i < len(traj)-1 else traj[i]

        obses.append(transition['obs'])
        next_obses.append(transition['new_obs'])
        actions.append(transition['action'])
        rewards.append(transition['reward'])
        not_dones.append(1.0 if i < len(traj) - 1 else 0.0)  # Last transition gets not_done = 0
        forces.append(-1.0 * transition['total_force'])
        force_vectors.append(torch.from_numpy(transition['cloth_force_vector']).float())

    n_step = 5
    discount_factors = args.discount_array
    discount_to_n_step_return, discount_to_future_return = compute_return_array(discount_factors, np.array(forces), n_step=n_step)
    for idx in range(len(traj) - n_step + 1):
        data_to_store_n_step_return = []
        for discount_factor in discount_factors:
            data_to_store_n_step_return.append(discount_to_n_step_return[discount_factor][idx])
            data_to_store_n_step_return.append(discount_to_future_return[discount_factor][idx])
        force_history_array = create_forces_history(forces)
        data_to_store = [ 
                obses[idx], force_history_array[idx], actions[idx].reshape(-1, 6)
            ]
        data_to_store[4:4] = data_to_store_n_step_return

    return {'obs': data_to_store[0], 'force_history': data_to_store[1], 'action': data_to_store[2], 'n_step_return_100': data_to_store[3], 'future_return_100': data_to_store[4]}


class ForceVisionDataset(Dataset):
    def __init__(self, split='train', data_root='data/dressing_proj/offline_force_data/test/', 
            discount=0.9,
            discount_factors=[0.9],
            use_delta_force=False,
            use_delta_mean=False,
            set_rotation_to_zero=False,
            use_weighted_loss=False,
            add_force_direction=False,
            exclude_garment=False,
            add_action_history=False,
            add_gripper_pose=False,
            convert_action_to_sim=False,
            subtract_threshold=False,
            force_history_length=5,
            *args, 
            **kwargs):

        super(ForceVisionDataset, self).__init__(*args, **kwargs)

        self.split = split
        self.scale_factor = 1
        self.voxel_size = 0.00625 * 10

        self.use_delta_force = use_delta_force
        self.use_delta_mean = use_delta_mean
        self.discount = discount
        self.set_rotation_to_zero = set_rotation_to_zero
        self.use_weighted_loss = use_weighted_loss
        self.add_force_direction = add_force_direction
        self.exclude_garment = exclude_garment
        self.add_action_history = add_action_history
        self.add_gripper_pose = add_gripper_pose
        self.convert_action_to_sim = convert_action_to_sim
        self.subtract_threshold = subtract_threshold
        self.force_history_length = force_history_length

        self.all_data_file = list_files(data_root)
        self.data_idxs = [i for i in range(len(self.all_data_file))]

        self.raw_data_list = {}

        for path in self.all_data_file:
            self.raw_data_list[path] = load_data_by_name(self.data_names, path)
        
        if self.use_weighted_loss:
            total_force = np.array([self.raw_data_list[key]['n_step_return_100'] for key in self.raw_data_list])
            force_sum = np.sum(np.abs(total_force)) 
        for path in self.all_data_file:
            if self.use_weighted_loss:
                self.raw_data_list[path]['force_weight'] = abs(self.raw_data_list[path]['n_step_return_100']) / force_sum * len(total_force)
            else:
                self.raw_data_list[path]['force_weight'] = 1

    def get(self, idx):
        return self.__getitem__(idx)

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        data_idx = self.data_idxs[idx]
        x = self.all_data_file[data_idx]
        raw_data = self.raw_data_list[x]
        obs = raw_data['obs']
        action = raw_data['action']

        if self.set_rotation_to_zero:
            action[:, 3] = 0
            action[: ,5] = 0
        if self.convert_action_to_sim:
            action[:, :3] = convert_sim_to_real(action[:, :3])
            action[:, 3:] = convert_sim_to_real(action[:, 3:])

        if self.exclude_garment:
            obs_pos = obs_pos[obs_x[:, 1] != 1]
            action = action[obs_x[:, 1] != 1]
            obs_x = obs_x[obs_x[:, 1] != 1]
            obs_x = obs_x[:, [0, 2]]
        
        # sim
        # n_step_return = raw_data['n_step_return_{}'.format(self.discount * 100)] * self.scale_factor
        # future_return = raw_data['future_return_no_{}'.format(self.discount * 100)] * self.scale_factor
        # real
        n_step_return = raw_data['n_step_return_{}'.format(self.discount * 100)] * self.scale_factor
        future_return = raw_data['future_return_{}'.format(self.discount * 100)] * self.scale_factor

        if self.subtract_threshold:
            n_step_return = np.minimum(0, n_step_return + 25)

        if self.use_delta_force:

            n_step_return = (- n_step_return) - np.sum(raw_data['force_history'][-5:])
            if self.use_delta_mean:
                n_step_return /= raw_data['force_history'].shape[0]

        if self.add_force_direction:
            force_direction_history = raw_data['force_direction_history']
        else:
            force_direction_history = np.zeros((15, ))

        if self.add_action_history:
            action_history = raw_data['action_history']
        else:
            action_history = np.zeros((15, ))

        if self.add_gripper_pose:
            ee_pose = raw_data['ee_pose'].flatten()
        else:
            ee_pose = np.zeros((7, ))

        others = {
            'force_history': torch.from_numpy(np.expand_dims(raw_data['force_history'][-self.force_history_length:].astype(np.float32) * self.scale_factor, axis=0)),
            # 'force_direction_history': torch.from_numpy(np.expand_dims(force_direction_history.astype(np.float32), axis=0)),
            # 'next_force_history': torch.from_numpy(np.expand_dims(raw_data['next_force_history'][-self.force_history_length:].astype(np.float32) * self.scale_factor, axis=0)),
            # 'action_history': torch.from_numpy(np.expand_dims(action_history.astype(np.float32), axis=0)),
            # 'ee_pose': torch.from_numpy(np.expand_dims(ee_pose.astype(np.float32), axis=0)),
            'action': torch.from_numpy(action.astype(np.float32)),
            'n_step_return': torch.from_numpy(np.expand_dims(np.asarray(n_step_return), axis=0)),
            # 'future_return': torch.from_numpy(np.expand_dims(np.asarray(future_return), axis=0)),
            # 'done': torch.from_numpy(np.expand_dims(np.asarray(raw_data['done']), axis=0)),
            # 'timestep': torch.from_numpy(np.expand_dims(np.asarray(raw_data['timestep']), axis=0)),
            # 'force_weight':  torch.from_numpy(np.expand_dims(np.asarray(raw_data['force_weight']), axis=0)),
        }
        
        return Data.from_dict(obs), Data.from_dict(others), x

    def get_class(self, n_step_return):
        force_sum = - n_step_return
        class_label = np.zeros(7)
        if force_sum < 25:
            class_label[0] = 1
        elif force_sum >= 25 and force_sum < 30:
            class_label[1] = 1
        elif force_sum >= 30 and force_sum < 35:
            class_label[2] = 1
        elif force_sum >= 35 and force_sum < 40:
            class_label[3] = 1
        elif force_sum >= 40 and force_sum < 45:
            class_label[4] = 1
        elif force_sum >= 45 and force_sum < 50:
            class_label[4] = 1
        elif force_sum >= 50:
            class_label[5] = 1
        return class_label

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.multiprocessing.set_start_method('spawn')
    train_dataset = ForceVisionDataset(data_root='/home/zhanyi/softagent_rpad/data/dressing_proj/0605_collect_person_1_training_trajectories/obs', discount=1, 
                                    discount_factors=[1], use_delta_force=False, 
                                    use_delta_mean=False, set_rotation_to_zero=False,
                                    use_weighted_loss=False,
                                    add_force_direction=False, exclude_garment=False, add_action_history=False, add_gripper_pose=False,
                                    convert_action_to_sim=False)
    
    valDataLoader = torch_geometric.loader.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0,
                                                  pin_memory=False, drop_last=True, follow_batch=['x'])
    new_comer_finetune_count = 0
    total_count = 1
    for iter in range(1):
        end = time.time()
        total_load_time = 0
        for idx, data in enumerate(valDataLoader):
            _, _, others, label = data
            if 'new_comer_finetune' in label[0]:
                new_comer_finetune_count += 1
            total_count += 1
            load_time = time.time() - end
            total_load_time += load_time

        print('iter {} data load time per batch: {}'.format(iter, total_load_time / idx))
