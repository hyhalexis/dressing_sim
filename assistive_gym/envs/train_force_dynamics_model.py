from collections import defaultdict
import copy
import json
import multiprocessing as mp
import os
import os.path as osp
import pickle
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
# import wandb
from logger import Logger

from chester import logger
import utils
from default_config import DEFAULT_CONFIG
from SAC_Q_function import ForceVisionAgent
from dressing_envs import DressingSawyerHumanEnv
from dataset import ForceVisionDataset


def list_files(rootdir):
    dirs = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


def update_env_kwargs(vv):
    new_vv = vv.copy()
    for v in vv:
        if v.startswith('env_kwargs_'):
            arg_name = v[len('env_kwargs_'):]
            new_vv['env_kwargs'][arg_name] = vv[v]
            del new_vv[v]
    return new_vv

class VArgs(object):
    def __init__(self, vv):
        for key, val in vv.items():
            setattr(self, key, val)

def vv_to_args(vv):
    args = VArgs(vv)

    # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    return args


def run_task(vv, log_dir=None, exp_name=None):
    # if resume from directory
    if vv['resume_from_ckpt']:
        log_dir = vv['resume_from_exp_path']
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir, exp_name=exp_name, format_strs=['csv'])
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)
    updated_vv = copy.copy(DEFAULT_CONFIG)
    updated_vv.update(**vv)

    main(vv_to_args(updated_vv))


def make_agent(obs_shape, action_shape, args, device, actor_load_name=None, force_critic_load_name=None):
    agent_class = ForceVisionAgent
    if args.add_force_direction:
        force_direction_history_shape = [15]
    else:
        force_direction_history_shape = [0]
    
    if args.add_action_history:
        action_history_shape = [15]
    else:
        action_history_shape = [0]

    if args.add_gripper_pose:
        ee_pose_shape = [7]
    else:
        ee_pose_shape = [0]
       
    return agent_class(
        args=args,
        obs_shape=obs_shape,
        action_shape=action_shape,
        force_history_shape=[5],
        force_direction_history_shape=force_direction_history_shape,
        action_history_shape=action_history_shape,
        ee_pose_shape=ee_pose_shape,
        device=device,
        hidden_dim=args.hidden_dim,
        discount=1,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        alpha_beta=args.alpha_beta,
        alpha_fixed=args.alpha_fixed,
        actor_lr=args.actor_lr,
        actor_beta=args.actor_beta,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_beta=args.critic_beta,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
        encoder_type=args.encoder_type,
        encoder_feature_dim=args.encoder_feature_dim,
        encoder_tau=args.encoder_tau,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
        log_interval=500,
        actor_load_name=actor_load_name,
        force_critic_load_name=force_critic_load_name,
        agent=args.agent,
    )

def train_q(agent, trainloader):
    running_loss_bootstrapping, running_loss_no_bootstrapping, running_loss_no_bootstrapping_l1 = 0.0, 0.0, 0.0

    for idx, data in enumerate(trainloader):
        loss_bootstrapping, loss_no_bootstrapping, loss_no_bootstrapping_l1 = agent.update_batch(data, idx)

        running_loss_bootstrapping += loss_bootstrapping.item()
        running_loss_no_bootstrapping += loss_no_bootstrapping.item()
        running_loss_no_bootstrapping_l1 += loss_no_bootstrapping_l1.item()

    return running_loss_bootstrapping/(idx+1), running_loss_no_bootstrapping/(idx+1), loss_no_bootstrapping_l1/(idx+1)

def eval_q(agent, evalloader):
    running_loss_bootstrapping, running_loss_no_bootstrapping, running_loss_no_bootstrapping_l1 = 0.0, 0.0, 0.0
    idx = 0
    for idx, data in enumerate(evalloader):
        loss_bootstrapping, loss_no_bootstrapping, loss_no_bootstrapping_l1 = agent.eval_batch(data)
        running_loss_bootstrapping += loss_bootstrapping.item()
        running_loss_no_bootstrapping += loss_no_bootstrapping.item()
        running_loss_no_bootstrapping_l1 += loss_no_bootstrapping_l1.item()

    if idx == 0:
        return 0, 0
    return running_loss_bootstrapping/(idx+1), running_loss_no_bootstrapping/(idx+1), loss_no_bootstrapping_l1/(idx+1)


def main(args):
    mp.set_start_method('forkserver', force=True)
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs

    args.encoder_type = args.encoder_type
    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)

    args.work_dir = logger.get_dir()
    # args.work_dir =  '/scratch/alexis/data/0302_dynamics_model_mag'
    os.makedirs(args.work_dir, exist_ok=True)

    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    print(model_dir)

    device = torch.device('cuda:{}'.format(args.cuda_idx) if torch.cuda.is_available() else 'cpu')

    action_shape = [6]
    obs_shape = (30000,)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))
    L = Logger(args.work_dir, use_tb=args.save_tb, chester_logger=logger)

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    offline_data_path_train = '/scratch/alexis/data/traj_data_with_one-hot_flex-75036/trajs'
    print(ForceVisionDataset.__module__)

    train_dataset = ForceVisionDataset(data_root=offline_data_path_train, discount=args.discount, 
                                    discount_factors=args.discount_array, use_delta_force=args.use_n_step_delta, 
                                    use_delta_mean=args.use_delta_mean, set_rotation_to_zero=args.set_rotation_to_zero,
                                    use_weighted_loss=args.use_weighted_loss, 
                                    add_force_direction=args.add_force_direction, exclude_garment=args.exclude_garment,
                                    add_action_history=args.add_action_history, add_gripper_pose=args.add_gripper_pose,
                                    convert_action_to_sim=args.convert_action_to_sim,
                                    subtract_threshold=args.subtract_threshold,
                                    force_history_length=args.force_history_length)

 
    trainloader = torch_geometric.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                                pin_memory=False, drop_last=False, follow_batch=['x'])

    num_epoch = args.num_epoch
    for epoch in range(num_epoch):
        agent.train()
        running_loss_bootstrapping, running_loss_no_bootstrapping, running_loss_no_bootstrapping_l1 = train_q(agent, trainloader=trainloader, use_iql=args.use_iql)
        if not args.use_iql:
            L.log('train_critic/loss_bootstrapping', running_loss_bootstrapping, epoch)
            L.log('train_critic/loss_no_bootstrapping', running_loss_no_bootstrapping, epoch)
            L.log('train_critic/loss_no_bootstrapping_l1', running_loss_no_bootstrapping_l1, epoch)
        else:
            L.log('train_critic/value_loss', running_loss_bootstrapping, epoch)
            L.log('train_critic/q_loss', running_loss_no_bootstrapping, epoch) 
        L.dump(epoch)

        if (epoch+1) % 100 == 0:
            agent.save(model_dir, epoch)
            

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()