from collections import defaultdict
# matplotlib.use('Agg')
import copy
import json
import multiprocessing as mp
import os
import os.path as osp
import pickle
import time
import datetime
import dateutil.tz
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import wandb
from chester.run_exp import run_experiment_lite, VariantGenerator

from torch_geometric.data import Batch
from reward_model import RewardModel3, RewardModelVLM
from chester import logger
import utils
from default_config import DEFAULT_CONFIG
from logger import Logger
from pc_replay_buffer import PointCloudReplayBuffer
from iql_pointcloud import IQLAgent
from visualization import save_numpy_as_gif
from dressing_envs import DressingSawyerHumanEnv
import pybullet as p

import multiprocessing as mp
from multiprocessing import Pool

# import cProfile, pstats, io

class ParallelEvaluator:
    """
    Parallel evaluator for running the `evaluate` function across multiple agents.
    """

    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.pool = Pool(processes=num_workers)

    def evaluate_agents(self, agent, step, pairs, args):
        task_args = [
            (agent, step, garment_id, motion_id, args)
            for garment_id, motion_id in pairs
        ]

        ret = self.pool.map(evaluate, task_args)
        return np.mean(ret)


    def close(self):
        """
        Clean up resources.
        """
        self.pool.close()
        self.pool.join()

### this function transforms the policy outputted action to be the action that is executable in the environment
# e.g., for dense transformation policy, this extracts the action corresponding to the gripper point, which is the last action in the action array
def create_vg_from_json(json_file):
    assert json_file is not None
    with open(json_file, 'r') as f:
        variant_dict = json.load(f)

    vg = VariantGenerator()
    for key, value in variant_dict.items():
        vg.add(key, [value])

    return vg

# plot some training info for debugging
def get_train_plot_image(rewards, imsize, ep_info, info):
    dpi = 100
    fig = plt.figure(figsize=(imsize/dpi, imsize/dpi), dpi=dpi)
    ax_total = fig.add_subplot(1, 1, 1) # plot total reward

    ax_total.plot(range(len(ep_info)), rewards, label='total_reward', color='C0')
    # ax_total.plot(range(len(ep_info)), [inf['angle_deviation_penalty'] for inf in ep_info], label='angle deviation penalty', color='C1')
    # ax_total.plot(range(len(ep_info)), [inf['pitch_rotation'] for inf in ep_info], label='pitch_rotation', color='C1')
    # ax_total.plot(range(len(ep_info)), [inf['yaw_rotation'] for inf in ep_info], label='yaw_rotation', color='C2')
    # ax_total.plot(range(len(ep_info)), [inf['translation_mag'] * 5e3 for inf in ep_info], label='translation_mag', color='C3')

    ax_total.legend()

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close("all")

    return image_from_plot

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
    # log_dir = "/home/sreyas/Desktop/Projects/softagent_rpad/data/reward_collection/"
    if vv.variants()[-1]['resume_from_ckpt']:
        log_dir = vv.variants()[-1]['resume_from_exp_path']

    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir, exp_name=exp_name, format_strs=['csv'])
    logdir = logger.get_dir()
    print("Logging at ", log_dir)
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)
    updated_vv = copy.copy(DEFAULT_CONFIG)
    vv = vv.variants()[-1]
    updated_vv.update(**vv)
    print("Logging at ", log_dir)
    if updated_vv['use_wandb']:
        updated_vv['wandb_group'] = None
        wandb.init()
        # wandb.init(project="dressing-flow", name=exp_name, entity="conan-iitkgp", config=updated_vv, settings=wandb.Settings(start_method='fork'))
    main(vv_to_args(updated_vv))
  

def get_info_stats(infos):
    all_keys = infos[0][0].keys()
    stat_dict = {}
    for key in all_keys:
        stat_dict[key + '_mean'] = []
        stat_dict[key + '_final'] = []
        for traj_idx, ep_info in enumerate(infos):
            for time_idx, info in enumerate(ep_info):
                stat_dict[key + '_mean'].append(info[key])
            stat_dict[key + '_final'].append(info[key])
        stat_dict[key + '_mean'] = np.mean(stat_dict[key + '_mean'])
        stat_dict[key + '_final'] = np.mean(stat_dict[key + '_final'])

    return stat_dict


# find the rb with shortest length
def check_rb_min_size(replay_buffers, rb_limit):
    min_len = 1e6
    for rb in replay_buffers:
        if rb.full:
            min_len = min(min_len, rb_limit)
        else:
            min_len = min(min_len, rb.idx)
    return min_len

# run evaluation of a policy
def evaluate(arg):
    agent, step, garment_id, motion_id, args = arg

    def run_eval_loop():
        env = DressingSawyerHumanEnv(policy=args.policy, horizon=args.horizon, camera_pos=args.camera_pos, occlusion=args.occlusion, use_force=args.use_force, one_hot=args.one_hot, reconstruct = args.reconstruct, render=args.render, gif_path=args.gif_path)

        obs, force_vector = env.reset(garment_id=garment_id, motion_id=motion_id, step_idx=step)
        done = False
        episode_reward = 0
        ep_info = []
        rewards = []
        
        t = 0
        while not done:
            print('------Iteration', t)

            with utils.eval_mode(agent):
                action = agent.select_action(obs, force_vector)
                # print('outside', force_vector.shape)
            step_action = action.reshape(-1, 6)[-1].flatten()
            # step_action[3] = 0

            # step_action[:3] *= 0.01
            x, y, z = step_action[:3].copy() * args.action_scale
            x_r, y_r, z_r = step_action[3:].copy()

            step_action[0] = z 
            step_action[1] = x
            step_action[2] = y

            # step_action[3] = z_r
            step_action[3] = 0
            step_action[4] = 0
            step_action[5] = y_r

            dtheta = np.linalg.norm(step_action[3:])
            max_rot_axis_ang = (5. * np.pi / 180.)

            if dtheta > 0:
                step_action[3:] = step_action[3:]/dtheta
                
                if dtheta > max_rot_axis_ang:
                    dtheta *= max_rot_axis_ang / np.sqrt(3)
                step_action[3:] *= dtheta

            obs, reward, done, info, force_vector = env.step(step_action)
            episode_reward += reward
            ep_info.append(info)
            rewards.append(reward)
            t += 1
        env.disconnect()
        return info['upperarm_ratio']
        
    upperarm_ratio = run_eval_loop()
    # L.log('eval/garment{}_motion{}_upperarm_ratio'.format(garment_id, motion_id), upperarm_ratio, step)
    # L.dump(step)
    return upperarm_ratio

def make_agent(obs_shape, action_shape, args, device, agent="IQL"):
    # create the sac training agent

    return IQLAgent(
        args=args,
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        hidden_dim=args.hidden_dim,
        discount=args.discount,
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
        encoder_lr=args.encoder_lr,
        encoder_tau=args.encoder_tau,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
        log_interval=args.log_interval,
        detach_encoder=args.detach_encoder,
        curl_latent_dim=args.curl_latent_dim,
        actor_load_name=args.actor_load_name,
        critic_load_name=args.critic_load_name,
        value_load_name=args.value_load_name,
        agent=args.agent,
        use_teacher_ddpg_loss=args.__dict__.get("use_teacher_ddpg_loss", False),
    )

def main(args):
    mp.set_start_method('forkserver', force=True)
    import tempfile
    tempfile.tempdir = '/scratch/alexis/tmp'
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs
    
    print('Horizon', args.horizon)

    if args.policy == 1:
        args.actor_load_name = args.actor_load_name_1
    
    elif args.policy == 2:
        args.actor_load_name = args.actor_load_name_2

    print('Policy: ', args.actor_load_name)
    if args.resume_from_ckpt:
        print('Resuming from: ', args.resume_from_path_actor, args.resume_from_path_critic, args.resume_from_path_value)

    # env = DressingSawyerHumanEnv(policy=args.policy, horizon=args.horizon, camera_pos=args.camera_pos, rand=args.rand, render=args.render, path_suffix=int(args.r1_w*10))
    gif_path = '{}/gifs'.format(logger.get_dir())
    os.makedirs(gif_path, exist_ok=True)
    print('Saving gifs at: ', gif_path)


    # make directory for logging
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    dataset_dir = args.dataset_dir
    args.work_dir = logger.get_dir()
    args.gif_path = gif_path
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))
    L = Logger(args.work_dir, use_tb=args.save_tb, chester_logger=logger)

    # create replay buffer
    # device = torch.device('cuda:{}'.format(args.cuda_idx) if torch.cuda.is_available() else 'cpu')
    device = 'cuda:0'
    action_shape = (6,)
    obs_shape = (30000,)
    replay_buffers = []
    rb_limit = args.replay_buffer_capacity // args.replay_buffer_num
 
    buffer = PointCloudReplayBuffer(
        args, action_shape, rb_limit, args.batch_size, device, td=args.__dict__.get("td", False), n_step=args.__dict__.get("n_step", 1),reward_relabel=args.reward_relabel
    )
    buffer.load2('/media/alexis/f8d6014a-8745-471e-ac53-5e50bf9ae322/alexis/traj_data_with_force_reconstr/trajs')

    reward_model1 = RewardModelVLM(obs_shape, action_shape, args, use_action=args.reward_model_use_action)
    print(args.pc_feature_dim)
    reward_model1.load(args.reward_model1_dir, args.reward_model1_step)
    reward_model1.eval()
    
    # reward_model2 = RewardModelVLM(obs_shape, action_shape, args, use_action=args.reward_model_use_action)
    # reward_model2.load(args.reward_model2_dir, args.reward_model2_step)
    # reward_model2.eval()

    if args.reward_relabel:
        with torch.no_grad():
            # buffer.relabel_rewards(reward_model1, reward_model2, device=device)
            buffer.relabel_rewards(reward_model1, None, device=device)
        print("Relabeling done")

    replay_buffers.append(buffer)
    # create agent
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    # parepare statistics
    best_avg_upper_arm_ratios_train, best_avg_upper_arm_ratios_unseen = -1e6, -1e6

    # if resume exp
    if args.resume_from_ckpt:
        agent.load(args.resume_from_path_actor, args.resume_from_path_critic, args.resume_from_path_value, load_optimizer=args.resume_from_ckpt)
        ckpt  = torch.load(osp.join(args.resume_from_path_critic), map_location=device)
        resume_step = ckpt['step']
        episode = ckpt['episode']
    else:
        resume_step = 0

    garment_ids = [1, 2]
    motion_ids = [0, 1, 2, 4, 5, 6, 7, 8]
    pairs = [(x, y) for x in garment_ids for y in motion_ids]
    print('Num workers', len(pairs))
        
    for step in tqdm(range(resume_step, args.num_train_steps)):
    # for i in range(3):
        # evaluate agent periodically
        # if True:
        #     step = i
        if step%args.eval_freq == 0 and step >= 0:
            episode = int(step/args.eval_freq)
            print("Eval begin step = {}".format(step))
            L.log('eval-number', episode, step)
            
            # print("Eval is happening")

            parallel_evaluator = ParallelEvaluator(num_workers=len(pairs))
            dressed_ratios = parallel_evaluator.evaluate_agents(agent, step, pairs, args)
            parallel_evaluator.close()

            L.log('eval/mean_upperarm_ratio', dressed_ratios, step)
            L.dump(step)

            if args.use_wandb: wandb.log({'mean_upper_arm_ratios': dressed_ratios, 'step': step})

            if dressed_ratios > best_avg_upper_arm_ratios_unseen:
                best_avg_upper_arm_ratios_unseen = dressed_ratios

                agent.save(model_dir, step, episode, is_best_test=True, best_avg_return_test=round(best_avg_upper_arm_ratios_unseen, 5))
           
            if args.save_model:
                agent.save(model_dir, step, episode)

            if args.save_buffer:
            # if False:
                for i in range(args.replay_buffer_num):
                    replay_buffers[i].save(buffer_dir, i)

            if args.evaluate_only:
                exit()

        # run training update
        agent.update(replay_buffers, L, step, pose_id=0)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
        
    exp_prefix =  '2025-0116-pybullet-from-scratch'
    load_variant_path = '/home/alexis/assistive-gym-film/assistive_gym/envs/variant_real_world_final_ori.json'
    
    loaded_vg = create_vg_from_json(load_variant_path)
    print("Loaded configs from ", load_variant_path)
    vg = loaded_vg
    print(vg)
    print('Number of configurations: ', len(vg.variants()))
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    exp_count = 0
    timestamp = now.strftime('%m_%d_%H_%M_%S')
    exp_name = "iql-training-film-force-simple"
    print(exp_name)
    exp_name = "{}-{}-{:03}".format(exp_name, timestamp, exp_count)
    log_dir = '/scratch/alexis/data/' + exp_prefix + "/" + exp_name

    run_task(vg, log_dir=log_dir, exp_name=exp_name)
    #eval()