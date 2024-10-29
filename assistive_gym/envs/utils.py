import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time
# from skimage.util.shape import view_as_windows
from collections import deque
import open3d as o3d
 


class ConvergenceChecker(object):
    def __init__(self, threshold, history_len):
        self.threshold = threshold
        self.history_len = history_len
        self.queue = deque(maxlen=history_len)
        self.converged = None

    def clear(self):
        self.queue.clear()
        self.converged = False

    def append(self, value):
        self.queue.append(value)

    def converge(self):
        if self.converged:
            return True
        else:
            losses = np.array(list(self.queue))
            self.converged = (len(losses) >= self.history_len) and (np.mean(losses[self.history_len // 2:]) > np.mean(
                losses[:self.history_len // 2]) - self.threshold)
            return self.converged


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def voxelize_pc(pc, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    try:
        voxelized_pcd = pcd.voxel_down_sample(voxel_size)
    except RuntimeError:
        return None
    voxelized_pc = np.asarray(voxelized_pcd.points)
    return voxelized_pc


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2 ** bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, image_size=84, transform=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self, use_curl=False):

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones, None


    def sample_cpc(self):

        start = time.time()
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

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs


def center_crop_image(image, output_size):
    # print('input image shape:', image.shape)
    if image.shape[0] ==1:
        image = image[0]
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top:top + new_h, left:left + new_w]
    # print('output image shape:', image.shape)
    return image


# import pouring.KPConv_sac.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
# import pouring.KPConv_sac.cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
# from pouring.KPConv_sac.utils import classification_inputs, PointCloudCustomBatch, grid_subsampling

def preprocess_single_obs(KPConv_model, obs, config, device, label_mean=None, label_std=None):
    obs_tmp = obs

    obs_ = obs_tmp[:, :3]
    # feature = obs_tmp[:, 3:]
    if config.first_subsampling_dl is not None:
        obs_subsampled = grid_subsampling(obs_, sampleDl=config.first_subsampling_dl)
    else:
        obs_subsampled = obs_

    tp_list = [obs_subsampled]
    stacked_points = np.concatenate(tp_list, axis=0)
    stack_lengths = np.array([tp.shape[0] for tp in tp_list], dtype=np.int32)
    
    # Input features
    stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)

    input_list = classification_inputs(config, 
                                        stacked_points,
                                        stacked_features,
                                        stack_lengths)

    obs_batch = PointCloudCustomBatch(input_list).to(device)
    with torch.no_grad():
        obs = KPConv_model(obs_batch).cpu().numpy()[0]

    if label_mean is not None:
        obs = obs * label_std + label_mean
        
    return obs

# visualize the reward line and its intersection with the garment triangles
def show_line_and_triangle(forearm_distance, upperarm_distance, img_width, img_height, cloth_particles, grasped_particles, 
    right_arm_particles, triangle, line_ori, line_dir, line2_ori, line2_dir, starting_point, gripper_pos, 
    intersection_point1, intersection_point2, observation_cloth_only, ang1=None, ang2=None, forward_distance=None):
    import matplotlib.pyplot as plt

    a, b, c, d, e, f = triangle
    if ang1 is not None:
        ang1 = round(ang1, 1)
        ang2 = round(ang2, 1)
        forward_distance = round(forward_distance, 2)

    mean_point = np.mean(triangle, axis=0)

    dpi = 100
    fig = plt.figure(figsize=(img_width/dpi, img_height/dpi), dpi=dpi)
    ax = plt.axes(projection='3d')

    # plot the polygons
    ax.plot([a[0], b[0]], [a[2], b[2]], [a[1], b[1]], color='blue', linewidth=2)
    ax.plot([b[0], c[0]], [b[2], c[2]], [b[1], c[1]], color='blue', linewidth=2)
    ax.plot([c[0], d[0]], [c[2], d[2]], [c[1], d[ 1]], color='blue', linewidth=2)
    ax.plot([d[0], e[0]], [d[2], e[2]], [d[1], e[ 1]], color='blue', linewidth=2)
    ax.plot([e[0], f[0]], [e[2], f[2]], [e[1], f[ 1]], color='blue', linewidth=2)
    ax.plot([f[0], a[0]], [f[2], a[2]], [f[1], a[ 1]], color='blue', linewidth=2)

    # plt.show()

    # plot the center of polygon and begining of the reward line
    ax.scatter(mean_point[0], mean_point[2], mean_point[1], color='green', linewidth=1.5)
    ax.scatter(starting_point[0], starting_point[2], starting_point[1], color='green', linewidth=1.5)
    ax.plot([mean_point[0], starting_point[0]], 
        [mean_point[2], starting_point[2]], 
        [mean_point[1], starting_point[1]], color='blue')

    # plt.show()

    # plot the human limb
    ax.plot([line_ori[0], starting_point[0] + line_dir[0] * 0.15], 
        [line_ori[2], starting_point[2] + line_dir[2] * 0.15], 
        [line_ori[1], starting_point[1] + line_dir[1] * 0.15], color='black')
    ax.plot([line2_ori[0], line_ori[0] + line2_dir[0] * 0.15], 
        [line2_ori[2], line_ori[2] + line2_dir[2] * 0.15], 
        [line2_ori[1], line_ori[1] + line2_dir[1] * 0.15], color='black')


    # plot voxelized cloth and human and gripper as observation
    ax.scatter(cloth_particles[:, 0], cloth_particles[:, 2], cloth_particles[:, 1], s=0.7, color='red')
    if not observation_cloth_only:
        ax.scatter(right_arm_particles[:, 0], right_arm_particles[:, 2], right_arm_particles[:, 1], s=0.7, color='black')
    gripper_pos = gripper_pos.reshape(-1, 3)
    ax.scatter(gripper_pos[:, 0], gripper_pos[:, 2], gripper_pos[:, 1], s=2, color='blue')

    # p1, p2 = shoulder_point1, shoulder_line_dir * 0.3 + shoulder_point1
    # # shoulder line
    # ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]], color='green')    
    # p1, p2 = shoulder_point1, arm_plane_norm * 0.3 + shoulder_point1
    # # norm to arm plane
    # ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]], color='red')

    # plot grasped particles
    ax.scatter(grasped_particles[:, 0], grasped_particles[:, 2], grasped_particles[:, 1], s=0.7, color='yellow')

    # plt.show()

    # plot the intersection point of the human line and the garment polygon
    if intersection_point2 is not None:
        # print("plotting upper arm intersection")
        ax.scatter(intersection_point2[0], intersection_point2[2], intersection_point2[1], color='green', linewidth=2)
        ax.scatter(line_ori[0], line_ori[2], line_ori[1], color='green', linewidth=2)
        ax.plot([intersection_point2[0], line_ori[0]], 
            [intersection_point2[2], line_ori[2]], 
            [intersection_point2[1], line_ori[1]], color='green')
    else:
        if intersection_point1 is not None:
            # print("plotting lower arm intersection")
            ax.scatter(intersection_point1[0], intersection_point1[2], intersection_point1[1], color='green', linewidth=2)
            ax.scatter(starting_point[0], starting_point[2], starting_point[1], color='green', linewidth=2)
            ax.plot([intersection_point1[0], starting_point[0]], 
                [intersection_point1[2], starting_point[2]], 
                [intersection_point1[1], starting_point[1]], color='green')


    # plot gripper local coordinate system
    gripper_pos = gripper_pos.flatten()

    ax.set_title('f: {} u: {}'.format(round(forearm_distance, 2), round(upperarm_distance, 2)))
    ax.view_init(azim=107, elev=30)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    #plt.show()
    #plt.cla()
    plt.close("all")

    return image_from_plot

# judge if a line intersects with a triangle
def line_intersecting_triangle(line_o, line_d, a, b, c):
    # line_o: origin of the line
    # line_d: direction of the line
    # a, b, c: three vertices of the triangle
    e1 = b - a
    e2 = c - a
    normal = np.cross(e1, e2)
    det = -np.dot(line_d, normal)
    try:
        invdet = 1.0 / det
    except ZeroDivisionError:
        return False, np.zeros(3)

    a0 = line_o - a
    da0 = np.cross(a0, line_d)
    u = np.dot(e2, da0) * invdet
    v = -np.dot(e1, da0) * invdet
    t = np.dot(a0, normal) * invdet
    intersect = (np.abs(det) >= 1e-7) and (t >= 0.0) and (u >= 0.0) and (v >= 0.0) and (u+v <= 1.0)
    intersection = line_o + line_d * t
    return intersect, intersection
    