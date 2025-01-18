from operator import itemgetter
from torch_geometric.data import Data, Batch
import numpy as np
import torch
import os.path as osp
import os
from utils import load_data_by_name
# from dressing_motion.dress_env import process_obs
from torch_geometric.data import Data
import scipy
import copy
import torch_geometric.transforms as T
from matplotlib import pyplot as plt
from tqdm import tqdm

def random_crop_pc(obs, action, max_x, min_x, max_y, min_y, max_z, min_z):
    gripper_pos = obs.pos[obs.x[:, 2] == 1]

    gripper_x_min, gripper_x_max = torch.min(gripper_pos[:, 0]).item(), torch.max(gripper_pos[:, 0]).item()
    gripper_y_min, gripper_y_max = torch.min(gripper_pos[:, 1]).item(), torch.max(gripper_pos[:, 1]).item()
    gripper_z_min, gripper_z_max = torch.min(gripper_pos[:, 2]).item(), torch.max(gripper_pos[:, 2]).item()


    x_start = np.random.rand() * (gripper_x_min - min_x) + min_x
    y_start = np.random.rand() * (gripper_y_min - min_y) + min_y
    z_start = np.random.rand() * (gripper_z_min - min_z) + min_z

    x_end = x_start + (max_x - min_x) * 0.75
    y_end = y_start + (max_y - min_y) * 0.75
    z_end = z_start + (max_z - min_z) * 0.75

    x_end = max(x_end, gripper_x_max)
    y_end = max(y_end, gripper_y_max)
    z_end = max(z_end, gripper_z_max)

    mask = (obs.pos[:, 0] <= x_end) & (obs.pos[:, 0] >= x_start) & \
            (obs.pos[:, 1] <= y_end) & (obs.pos[:, 1] >= y_start) & \
            (obs.pos[:, 2] <= z_end) & (obs.pos[:, 2] >= z_start)

    obs.pos = obs.pos[mask]
    obs.x = obs.x[mask]
    obs.batch = obs.batch[mask]

    return obs, action[mask]

def rotate_pc(obs, angle, device):
    angles = np.random.uniform(-angle, angle, size=3)
    Rx = np.array([[1,0,0],
                [0,np.cos(angles[0]),-np.sin(angles[0])],
                [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                [0,1,0],
                [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                [np.sin(angles[2]),np.cos(angles[2]),0],
                [0,0,1]])

    R = np.dot(Rz, np.dot(Ry,Rx))
    R = torch.from_numpy(R).to(self.device).float()
    obs.pos = obs.pos @ R
    return obs

def scale_pc(obs, scale_low, scale_high):
    s = np.random.uniform(scale_low, scale_high)
    obs.pos = obs.pos * s
    return obs

class PointCloudReplayBuffer():
    """Buffer to store environment transitions."""

    def __init__(self, args, action_shape, capacity, batch_size, device, transform=None,
        data_parallel=False, td=False, discount=0.99, n_step=3, reward_relabel=False):

        self.args = args
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.transform = transform
        self.data_parallel = data_parallel
        self.td = td
        self.discount = discount
        self.n_step = n_step
        self.round = 0

        self.obses = [_ for _ in range(self.capacity)]
        self.next_obses = [_ for _ in range(self.capacity)]
        if 'flow' in self.args.encoder_type:
            if self.args.action_mode == 'translation':
                self.actions = [_ for _ in range(self.capacity)]
            elif self.args.action_mode == 'rotation':
                if self.args.flow_q_mode == 'repeat_se3':
                    self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
                elif self.args.flow_q_mode in ['all_flow', 'repeat_gripper', 'concat_latent']:
                    self.actions = [_ for _ in range(self.capacity)]
        else:
            self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        

        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.reward_relabel = reward_relabel

        self.rewards_pref = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.label_intersection_idxes = np.empty((capacity, 1), dtype=np.int64)
        self.non_randomized_obses = [_ for _ in range(self.capacity)]
        self.next_non_randomized_obses = [_ for _ in range(self.capacity)]
        self.gripper_idx = args.gripper_idx
        self.idx = 0
        self.last_save = 0
        self.full = False
        # self.buffer_start = 0
        self.rotation_transform = T.RandomRotate(360, 1)
        self.scale_transform = T.RandomScale((0.8, 1.2))
        self.reward1_w = args.r1_w
        self.reward2_w = args.r2_w

    def add(self, obs, action, reward, next_obs, done, non_randomized_obs=None, next_non_randomized_obs=None, label_intersection_idx=-1):
        # NOTE: now: obs is actually a data object from pytorch geometric
        # print("obs.x.shape: ", obs.x.shape)
        self.obses[self.idx] = obs
        self.next_obses[self.idx] = next_obs
        # print(obs.x.shape, action.shape)
        # print("Encoder type: ", self.args.encoder_type)
        if 'flow' in self.args.encoder_type:
            self.actions[self.idx] = action
        else:
            np.copyto(self.actions[self.idx], action)

        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.label_intersection_idxes[self.idx], label_intersection_idx)
        # print(self.label_intersection_idxes[:self.idx])

        if self.args.full_obs_guide:
            # print("adding nonrandomized obs")
            self.non_randomized_obses[self.idx] = non_randomized_obs
            # self.next_non_randomized_obses[self.idx] = next_non_randomized_obs

        # if self.buffer_start == 0:
        #     self.idx = (self.idx + 1) % self.capacity
        # else:
        #     self.idx = (self.idx + 1) % self.capacity
        #     if self.idx == 0:
        #         self.idx = self.buffer_start

        # self.full = self.full or self.idx == self.buffer_start

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self, curl=False, auxiliary=False):
        upper_limit = self.capacity if self.full else self.idx
        idxs = np.random.randint(
            0, upper_limit, size=self.batch_size
        )
        # idxs = torch.tensor(idxs, dtype=torch.long, device=self.device)

        # print(upper_limit, self.idx, idxs.shape)
        obses = list(itemgetter(*idxs)(self.obses))
        obses = [obs.to(self.device) for obs in obses]

        # for i in range(len(obses)):
        #     print("single observation shape: ", self.obses[i].x.shape, self.obses[i].pos.shape)
        next_obses = list(itemgetter(*idxs)(self.next_obses))
        next_obses = [obs.to(self.device) for obs in next_obses]

        obses = Batch.from_data_list(obses).to(self.device)
        # print("obs.x.shape: ", obses.pos.shape, obses.x.shape)
        next_obses = Batch.from_data_list(next_obses).to(self.device)
        rewards = np.array(self.rewards)
        not_dones = np.array(self.not_dones)
        rewards_pref = np.array(self.rewards_pref)
        force_vectors = list(itemgetter(*idxs)(self.force_vectors))
        all_tensors = all(isinstance(el, torch.Tensor) for el in force_vectors)
        if not all_tensors:
            print(force_vectors)
        force_vectors = torch.stack(force_vectors).to(self.device)
        
        rewards = torch.as_tensor(rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(not_dones[idxs], device=self.device)
        reward_pref = torch.as_tensor(rewards_pref[idxs], device=self.device)
        # force_vectors = torch.as_tensor(force_vectors[idxs], device=self.device)
        # print("rewards shape: ", rewards.shape)
        # print("not_dones shape: ", not_dones.shape)

        if self.td: 
            for i in range(1, self.n_step):
                not_dones_accu = not_dones.clone()
                for j in range(1, i):
                    not_dones_current = torch.as_tensor(self.not_dones[idxs + j], device=self.device)
                    not_dones_accu = torch.logical_and(not_dones_accu, not_dones_current)
                step_reward = not_dones_accu * torch.as_tensor(self.rewards[idxs+i], device=self.device)
                step_reward_pref = not_dones_accu * torch.as_tensor(self.rewards_pref[idxs+i], device=self.device)
                # print("step reward shape: ", step_reward.shape)
                rewards += (self.discount ** i) * step_reward
                reward_pref += (self.discount ** i) * step_reward_pref
        
        if 'flow' in self.args.encoder_type:
            if self.args.action_mode == 'translation' or (self.args.action_mode == 'rotation' and self.args.flow_q_mode in ['all_flow', 'repeat_gripper', 'concat_latent']):
                actions = list(itemgetter(*idxs)(self.actions))
                actions = np.concatenate(actions)
                # print("sample proprio ---actions shape: ", actions.shape)
                actions = torch.from_numpy(actions).to(self.device)
            else:
                actions = torch.as_tensor(self.actions[idxs], device=self.device)
        else:
            # print(self.args.encoder_type)
            actions = torch.as_tensor(self.actions[idxs], device=self.device)

        auxiliary_term = {}
        if auxiliary: 
            # print("sample non randomized obs")
            if self.non_randomized_obses[0] is None:
                auxiliary_term = {
                    'non_randomized_obses': None,
                    'reward_predictor_target': None,
                    'reward_predictor_obs': None,
                }
            else:

                non_randomized_obses = list(itemgetter(*idxs)(self.non_randomized_obses))
                non_randomized_obses = Batch.from_data_list(non_randomized_obses).to(self.device)

                label_intersection_idxes = self.label_intersection_idxes[idxs]
                good_idxes = (label_intersection_idxes != -1)
                label_intersection_idxes = label_intersection_idxes[good_idxes]

                auxiliary_term = {
                        'non_randomized_obses': non_randomized_obses,
                        'reward_predictor_target': None,
                        'reward_predictor_obs': None,
                }

                if len(label_intersection_idxes) >= 2 and self.args.train_reward_predictor:
                    good_idxes = np.array(good_idxes).astype(np.bool8).flatten()
                    reward_obses = list(itemgetter(*(idxs[good_idxes]))(self.obses))
                    obs_lengths = [obs.x.shape[0] for obs in reward_obses]
                    reward_obses = Batch.from_data_list(reward_obses).to(self.device)

                    target = np.zeros(reward_obses.x.shape[0])
                    start_idx = 0
                    for i in range(len(label_intersection_idxes)):
                        target[start_idx + label_intersection_idxes[i]] = 1
                        start_idx += obs_lengths[i]

                    auxiliary_term = {
                        'non_randomized_obses': non_randomized_obses,
                        'reward_predictor_target': torch.from_numpy(target).float().unsqueeze(1).to(self.device),
                        'reward_predictor_obs': reward_obses,
                    }

        curl_kwargs = {}
        if curl:
            if 'rotation' in self.args.augmentation_func:
                pos_len = obses.pos.shape[0]
                obses_positive = obses.detach().clone()
                obses_positive.pos = torch.cat([obses_positive.pos, obses_positive.pos + actions[:, :3]], dim=0)
                obses_positive.x = torch.cat([obses_positive.x, obses_positive.x], dim=0)
                obses_positive.batch = torch.cat([obses_positive.batch, obses_positive.batch])
                obses_positive = self.rotation_transform(obses_positive)

                actions_positive = actions.detach().clone()
                actions_positive[:, :3] = obses_positive.pos[pos_len:] - obses_positive.pos[:pos_len]

                obses_positive.pos = obses_positive.pos[:pos_len]
                obses_positive.x = obses_positive.x[:pos_len]
                obses_positive.batch = obses_positive.batch[:pos_len]

            if 'scale' in self.args.augmentation_func:
                pos_len = obses.pos.shape[0]
                obses_positive = obses.detach().clone()
                obses_positive.pos = torch.cat([obses_positive.pos, actions[:, :3], actions[:, 3:]], dim=0)
                obses_positive.x = torch.cat([obses_positive.x, obses_positive.x, obses_positive.x], dim=0)
                obses_positive.batch = torch.cat([obses_positive.batch, obses_positive.batch, obses_positive.batch])
                obses_positive = self.scale_transform(obses_positive)
                
                actions_positive = actions.detach().clone()
                actions_positive[:, :3] = obses_positive.pos[pos_len: 2*pos_len]
                actions_positive[:, 3:] = obses_positive.pos[2*pos_len: ]

                obses_positive.pos = obses_positive.pos[:pos_len]
                obses_positive.x = obses_positive.x[:pos_len]
                obses_positive.batch = obses_positive.batch[:pos_len]

            curl_kwargs = dict(
                obs_anchor=obses, obs_pos=obses_positive,
                action_anchor=actions, action_pos=actions_positive,
                time_anchor=None, time_pos=None
            )
        if self.reward_relabel:
            # print("Returning rewards_pref")
            return obses, actions, reward_pref, next_obses, not_dones, force_vectors, curl_kwargs, auxiliary_term
        return obses, actions, rewards, next_obses, not_dones, force_vectors, curl_kwargs, auxiliary_term


    def sample_cpc(self):
        # not used
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)
        pos = random_crop(pos, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def get_obs_from_stored_data(self, data):
        cloth_points = data['cloth_pc']
        object_points = data['object_pc']
        picker_pos = data['picker_pos']
        picker_pos = picker_pos.reshape((-1, 3))
        points, categories,voxelized_cloth = process_obs(object_points, cloth_points, picker_pos, self.args.voxel_size, return_voxel_cloth=True)
        
        obs = {
            'x': torch.from_numpy(categories).float(),
            'pos': torch.from_numpy(points).float(),
        }
        obs = Data.from_dict(obs)

        return obs, voxelized_cloth

    def fill_buffer(self, buffer_dir, fix_load_data=False):
        print("adding demo data from dir: ", buffer_dir)
        data_names = ['cloth_pc', 'object_pc', 'picker_pos', 'action', 'task_reward', 'force_reward', 'particle_pos']
        for split in ['train', 'valid']:
            data_root = osp.join(buffer_dir, split)
            all_traj_file = os.listdir(data_root)
                
            all_traj_file = sorted(all_traj_file)
            if 'meta_info.json' in all_traj_file:
                all_traj_file.remove("meta_info.json")

            for idx, traj_file in enumerate(all_traj_file):
                print("traj {}/{}".format(idx, len(all_traj_file)), flush=True)
                traj_path = osp.join(data_root, traj_file)
                traj_data = os.listdir(traj_path)
                if 'gif.gif' in traj_data:
                    traj_data.remove('gif.gif')
                
                traj_data = sorted(traj_data)
                for idx in range(len(traj_data[:-1])):
                    if idx % 50 == 0:
                        print("idx: ", idx, flush=True)
                    data_path = osp.join(data_root, traj_file, traj_data[idx])
                    next_data_path = osp.join(data_root, traj_file, traj_data[idx + 1])

                    data = load_data_by_name(data_names, data_path)
                    next_data = load_data_by_name(data_names, next_data_path)

                    obs, voxelized_cloth_pc = self.get_obs_from_stored_data(data)
                    next_obs, _ = self.get_obs_from_stored_data(next_data)
                    
                    if 'flow' in self.args.encoder_type:
                        cur_particle_pos = data['particle_pos']
                        next_particle_pos = next_data['particle_pos']
                        pc_to_particle_dist = scipy.spatial.distance.cdist(voxelized_cloth_pc, cur_particle_pos)
                        mapping_pc_to_particle = np.argmin(pc_to_particle_dist, axis=1)
                        cloth_particle_flow = next_particle_pos[mapping_pc_to_particle] - cur_particle_pos[mapping_pc_to_particle]

                        picker_pos, next_picker_pos = data['picker_pos'], next_data['picker_pos']
                        categories = obs.x.numpy()
                        action = np.zeros((categories.shape[0], 3))
                        object_num = np.sum(categories[:, 0] == 1)
                        cloth_num = np.sum(categories[:, 1] == 1)
                        action[object_num:object_num+cloth_num] = cloth_particle_flow
                        action[-1] = next_picker_pos.reshape(-1, 3) - picker_pos.reshape(-1, 3)
                        action = action * 100
                        action = action.astype(np.float32)
                    else:
                        action = data['action'].reshape(-1)[:3].reshape(-1, 3).astype(np.float32)

                    reward = data['task_reward'] + data['force_reward'] * self.args.force_w
                    done = False

                    self.add(obs, action, reward, next_obs, done)

        # if fix_load_data:
        #     self.buffer_start = self.idx
        print("adding buffer done!")


    def save(self, save_dir, rb_idx):
        if self.idx == self.last_save:
            return
        save_dir = save_dir + '/' + str(rb_idx)
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if self.last_save > self.idx:
            self.save_with_start_end(save_dir, self.last_save, self.capacity)
            self.round += 1
            self.save_with_start_end(save_dir, 0, self.idx)
        else:
            self.save_with_start_end(save_dir, self.last_save, self.idx)

        self.last_save = self.idx

    def save_with_start_end(self, save_dir, start, end):
        path = os.path.join(save_dir, '%d_%d_%d.pt' % (start, end, self.round))
        
        payload = [
            self.obses[start:end],
            self.next_obses[start:end],
            self.actions[start:end],
            self.rewards[start:end],
            self.not_dones[start:end],
            self.non_randomized_obses[start:end],
        ]

        self.last_save = self.idx
        torch.save(payload, path)
        print(f'Saved replay buffer to {path}')

    def parse_data(data_dir):
        import pickle5 as pickle
        file_nos = os.listdir(data_dir)
        # file_nos.remove("videos")
        file_nos = sorted(file_nos)
        file_nos = [os.path.join(data_dir, file_no) for file_no in file_nos]
        print(len(file_nos))
        trajs = []
        for i in tqdm(range(len(file_nos))):
            print("Loading file ", file_nos[i])
            file = file_nos[i]
            with open(file, 'rb') as f:
                data = pickle.load(f)
            trajs.append(data)

        transitions = [transition for traj in trajs for transition in traj]

        obses = torch.tensor([t['obs'] for t in transitions], dtype=torch.float32)
        next_obses = torch.tensor([t['new_obs'] for t in transitions], dtype=torch.float32)
        actions = torch.tensor([t['action'] for t in transitions], dtype=torch.float32)
        rewards = torch.tensor([t['reward'] for t in transitions], dtype=torch.float32)
        not_dones = torch.tensor([1 for t in transitions], dtype=torch.float32)
        non_randomized_obses = torch.tensor([t['obs'] for t in transitions], dtype=torch.float32)

        payload = [obses, next_obses, actions, rewards, not_dones, non_randomized_obses]
        return payload
    
    def save_as_payload(self, data_dir):
        import pickle5 as pickle
        import random
        file_nos = os.listdir(data_dir)
        # file_nos = random.sample(file_nos, len(file_nos) // 20)
        # file_nos.remove("videos")
        file_nos = sorted(file_nos)
        # file_nos = [file for file in file_nos if not file.startswith("p2_")]
        file_nos = [os.path.join(data_dir, file_no) for file_no in file_nos]
        # sampled_files = random.sample(file_nos, min(200, len(file_nos)))


        obses = []
        next_obses = []
        actions = []
        rewards = []
        not_dones = []
        non_randomized_obses = []
        forces = []
        ori_obses = []
        force_vectors = []

        for file in tqdm(file_nos, desc="Loading trajectory files"):
            # print("Loading file ", file)

            # with open(file, 'rb') as f:
            #     traj = pickle.load(f)

            try:
                with open(file, 'rb') as f:
                    traj = pickle.load(f)
                    # Process the loaded trajectory as needed
                    print(f"Successfully loaded {file}")
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"Skipping corrupted file: {file}. Error: {e}")
            except Exception as e:
                print(f"Unexpected error with file {file}: {e}")

            for i in range(len(traj)):
                # Add transition components directly to lists
                transition = traj[i]
                # next_transition = traj[i+1] if i < len(traj)-1 else traj[i]

                obses.append(transition['obs'])
                next_obses.append(transition['new_obs'])

                # ori_obses.append(transition['obs'])

                # obs = copy.deepcopy(transition['complete_pts'])
                # obses.append(obs)

                # new_obs = copy.deepcopy(next_transition['complete_pts'])
                # new_obs.x = torch.cat((new_obs.x, torch.zeros(new_obs.x.size(0), 1)), dim=1)
                # new_obs.x[-1, 3] = next_transition['total_force']
                # next_obses.append(new_obs)

                actions.append(transition['action'])
                # flex_action = transition['action'][[1, 2, 0, 4, 5, 3]]
                # actions.append(flex_action)
                rewards.append(transition['reward'])
                not_dones.append(1.0 if i < len(traj) - 1 else 0.0)  # Last transition gets not_done = 0
                non_randomized_obses.append(None)
                forces.append(transition['total_force'])
                if np.isscalar(transition['cloth_force_vector']) and transition['cloth_force_vector']  == 0:
                    transition['cloth_force_vector'] = np.zeros(3)
                force_vectors.append(torch.from_numpy(transition['cloth_force_vector'][[1, 2, 0]]).float())
        
        payload = [obses, next_obses, actions, rewards, not_dones, non_randomized_obses, forces, ori_obses, force_vectors]
        torch.save(payload, '/scratch/alexis/data/payload_film_force_simple_tiny')

    def load2(self, data_dir):
        # self.save_as_payload(data_dir)
        payload = torch.load('/scratch/alexis/data/payload_film_force_simple')

        # Convert lists to tensors
        self.obses = payload[0]
        self.next_obses = payload[1]
        self.ori_actions = [action.reshape(-1, 6) for action in payload[2]]
        temp = []
        for idx in range(len(payload[2])):
            obs_size = self.obses[idx].x.shape[0]
            action = payload[2][idx].reshape(-1, 6)[-1]
            actions = np.tile(action, (obs_size, 1))
            temp.append(actions)
        self.actions = temp
        # self.actions = [action.reshape(-1, 6) for action in payload[2]]
        self.rewards = payload[3]
        self.not_dones = payload[4]
        self.non_randomized_obses = payload[5]
        self.forces = payload[6]
        self.ori_obses = payload[7]
        self.force_vectors = payload[8]
        c = 0
        for f in self.force_vectors:
            if not isinstance(f, torch.Tensor):
                print(f)
                c+=1
        print(c)
        
        

        # temp = []
        # actions_gripper = payload[2]

        # for idx in range(len(actions_gripper)):
        #     obs_size = self.obses[idx].x.shape[0]
        #     actions = np.vstack([actions_gripper[idx]] * obs_size)
        #     temp.append(actions)
        
        # self.actions = temp
        self.idx = len(payload[3])
        self.last_save = self.idx


    def load(self, save_dir, rb_idx, fix_load_data=False):
        save_dir = save_dir + '/' + str(rb_idx)
        chunks = os.listdir(save_dir)
        for chunk in chunks: print(chunk.split('_'), int(chunk.split('_')[2][:-3]))
        chucks = sorted(chunks, key=lambda x: (- int(x.split('_')[2][:-3]), int(x.split('_')[0])))
        pre_round = 0
        for (i, chunk) in enumerate(chucks):
            start, end, round = [int(x) if 'pt' not in x else int(x[:-3]) for x in chunk.split('.')[0].split('_')]
            # print(round)
            if i == 0:
                self.round = round
            # this is a bit problematic as a small amount of newer data might be overwritten by older data, but this prevents empty slots in the replay buffer
            # print(chunk)
            # print('self_idx: {} end: {} round: {}'.format(self.idx, end, round))
            if self.idx > end and pre_round - 1 == round:
                continue
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            # assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.non_randomized_obses[start:end] = payload[5]
            print("Round: ", round, "start: ", start, "end: ", end, "idx: ", self.idx)
            if self.round == round:
                self.idx = end
            pre_round = round
            print(f'Loaded replay buffer from {path}')
            print("current idx {}".format(self.idx))
            if end == self.capacity: break
        self.last_save = self.idx
        # fix_load_data is always false for now
        # if fix_load_data:
        #     self.buffer_start = end
    
    
    def process(self, obs_1, act_1,device='cuda',debug=False):
        self.debug = debug
        if self.debug:
            print("process - Size of inputs as obs_1:", len(obs_1))
            print("process - Size of inputs as act_1:", len(act_1))
        # obs_1 = copy.deepcopy(obs_1)
        # obs_2 = copy.deepcopy(obs_2)
        # act_1 = copy.deepcopy(act_1)
        # act_2 = copy.deepcopy(act_2)
        if isinstance(obs_1[0], Data):
            # print("process - obs_1[0].x.shape:", obs_1[0].x.shape)
            obs_1 = Batch.from_data_list(obs_1).to(self.device)
            # print(obs_1)
        else:
            raise ValueError("obs_1 and obs_2 must be lists of Data objects.")
        
        if 'flow' in self.args.encoder_type:
            if self.args.action_mode == 'translation' or (self.args.action_mode == 'rotation' and self.args.flow_q_mode in ['all_flow', 'repeat_gripper', 'concat_latent']):
                # act_1 = np.concatenate(act_1)
                act_1 = act_1[0]# act_1 = np.concatenate(act_1)
                act_1 = torch.from_numpy(act_1).float().to(self.device)
        else:
                act_1 = act_1[0]
                act_1 = torch.as_tensor(act_1, device=self.device)
        
        if self.debug:
            print("process - Size after making into batch class obs1.x:", obs_1.x.shape)
            print("process - Size after making into concat act1:", act_1.shape)
            print("process - Batch size of obs1:", obs_1.batch)
        
        gripper_idx = 1 # reward model idx
        for b_idx in obs_1.batch:
            act_1[(obs_1.batch == b_idx)] = act_1[(obs_1.batch == b_idx) & (obs_1.x[:, gripper_idx] == 1)]
        return obs_1, act_1
    
    def relabel_rewards(self, reward_model1, reward_model2, device='cuda'):
        for i in tqdm(range(self.idx)):
            obs, action, reward, done, next_obs = self.obses[i], self.actions[i], self.rewards[i], self.not_dones[i], self.next_obses[i]
            # obs = ori_obs.to(self.device)
            
            obs = obs.to(self.device)

            # obs_force = copy.deepcopy(ori_obs)
            # obs_force = obs_force.to(self.device)
            # obs_force.x = torch.cat((obs_force.x, torch.zeros(ori_obs.x.size(0), 1, device=device)), dim=1)
            # obs_force.x[-1, 3] = force
            # ori_obs = obs_force.to(self.device)

            obs, act = self.process([obs], [action], device=device)
            reward = reward
            done = done
            with torch.no_grad():
                if reward_model2 is None:  
                    reward_pref = reward_model1.r_hat(obs, act)
                else:
                    reward_pref = self.reward1_w * reward_model1.r_hat(obs, act) + self.reward2_w * reward_model2.r_hat(obs, act)
            self.rewards_pref[i] = reward_pref