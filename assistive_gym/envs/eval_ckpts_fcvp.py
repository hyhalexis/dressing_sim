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
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import wandb
from chester.run_exp import run_experiment_lite, VariantGenerator
from collections import deque

from torch_geometric.data import Batch
from chester import logger
import utils
from default_config import DEFAULT_CONFIG
from logger import Logger
from iql_pointcloud import IQLAgent
from dressing_envs import DressingSawyerHumanEnv

import multiprocessing as mp
from multiprocessing import Pool
from train_force_dynamics_model import make_agent

# import cProfile, pstats, io

class ParallelEvaluator:
    """
    Parallel evaluator for running the `evaluate` function across multiple agents.
    """

    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.pool = Pool(processes=num_workers)

    def evaluate_agents(self, agent, step, elbow_rand, shoulder_rand, pairs, args):
        task_args = [
            (agent, step, elbow_rand, shoulder_rand, garment_id, motion_id, pose_id, args)
            for garment_id, motion_id, pose_id in pairs
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
    agent, step, elbow_rand, shoulder_rand, garment_id, motion_id, pose_id, args = arg
    # if step == 0:
    #     reconstruct = True
    # else:
    #     reconstruct = False

    # if step <= 1:
    #     use_force = True
    # else:
    #     use_force = False


    def run_eval_loop():
        # Note
        # env = DressingSawyerHumanEnv(policy=args.policy, horizon=args.horizon, camera_pos=args.camera_pos, occlusion=args.occlusion, rand=args.rand, render=args.render, gif_path=args.gif_path)
        env = DressingSawyerHumanEnv(policy=int(step), horizon=args.horizon, camera_pos=args.camera_pos, occlusion=args.occlusion, render=args.render, gif_path=args.gif_path, one_hot=args.one_hot, reconstruct=args.reconstruct, use_force=args.use_force, elbow_rand=elbow_rand, shoulder_rand=shoulder_rand)

        obs, force_vector = env.reset(garment_id=garment_id, motion_id=motion_id, pose_id=pose_id, step_idx = step)
        print('forceee', force_vector)
        done = False
        episode_reward = 0
        ep_info = []
        rewards = []

        force_history = deque([], maxlen=5)
        for i in range(5):
            force_history.append(0)
        force_history.append(force_vector.sum())
        
        t = 0
        while not done:
            print('------Iteration', t)

            with utils.eval_mode(agent):
                if env.on_forearm or env.on_upperarm:
                    action, progression_direction, predicted_force = agent.plan_action_constrain_force(obs, force_vector, np.array(force_history))
                else:
                    action = agent.sample_action(obs, force_vector)
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
            force_history.append(force_vector.sum())
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

def main(args):
    mp.set_start_method('forkserver', force=True)
    import tempfile
    tempfile.tempdir = '/scratch/alexis/tmp'
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs
    
    print('Horizon', args.horizon)
    print('Occlusion?', args.occlusion)

    # make directory for logging
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)

    # create replay buffer
    # device = torch.device('cuda:{}'.format(args.cuda_idx) if torch.cuda.is_available() else 'cpu')
    device = 'cuda:0'
    action_shape = (6,)
    obs_shape = (30000,)
        
     # Note
    dir = logger.get_dir()

    # dir = os.path.dirname(folder_path)

    L = Logger(dir, use_tb=args.save_tb, chester_logger=logger)

    gif_path = '{}/gifs'.format(dir)
    os.makedirs(gif_path, exist_ok=True)
    print('Saving gifs at: ', gif_path)
    args.gif_path = gif_path

    agents_ckpts = [
        "/scratch/alexis/data/1223_flex_one-hot/model/actor_best_test_750137_0.75036.pt",

                    # "/scratch/alexis/data/2025-0104-pybullet-from-scratch/iql-training-from-scratch-force-reconstr-01_05_09_01_48-000/model/actor_best_test_400000_0.88345.pt",
                    # "/scratch/alexis/data/2025-0110-pybullet-from-scratch/iql-training-from-scratch-force-reconstr-reduced-data-01_11_06_42_34-000/model/actor_best_test_280000_0.82393.pt"
                    # "/scratch/alexis/data/2024-1220-pybullet-from-scratch/iql-training-from-scratch-with-force-p1-only-12_20_16_20_55-000/model/actor_best_test_80000_0.91073.pt",
                    # "/home/alexis/assistive-gym-film/assistive_gym/envs/ckpt/actor_1900106.pt",

                    # "/scratch/alexis/data/2024-1205-pybullet-finetuning/iql-training-p1-reward_model1-only-12_05_07_16_53-000/model/actor_60000.pt",
                    # "/scratch/alexis/data/2024-1205-pybullet-finetuning/iql-training-p1-reward_model1-only-12_05_07_16_53-000/model/actor_220000.pt",
                    # "/scratch/alexis/data/2024-1205-pybullet-finetuning/iql-training-p2-reward_model1-only-12_05_05_50_38-000/model/actor_best_test_20000_0.74911.pt"
                    ]

    # agents_ckpts = ["/home/alexis/assistive-gym-film/assistive_gym/envs/ckpt_rss/actor_160111.pt"]
         
    # Note
    agents = []
    import re

    for idx in range(10):
        rng = np.random.default_rng()
        if idx == 0:
            elbow_rand = -90
            shoulder_rand = 80
        else:
            elbow_rand = np.round(-90 + rng.uniform(-10, 10), 2)
            shoulder_rand = np.round(80 + rng.uniform(-5, 5), 2)

        i = 0
        for ckpt in agents_ckpts:
            if i == 0:
                args.pc_feature_dim = 3
                args.use_film = False
            # elif i == 1:
            #     args.pc_feature_dim = 4
            # else:
            #     args.pc_feature_dim = 3


            agent = make_agent(
                obs_shape=obs_shape,
                action_shape=action_shape,
                args=args,
                device=device,
                actor_load_name=ckpt,
                force_critic_load_name='/home/alexis/softagent_rpad/data/local/031724_train_force_dynamics_model_region_13_reproduce/031724_train_force_dynamics_model_region_13_reproduce-03_02_22_18_31-001/model/force_critic_2000.pt'
            )
            # agent.load_actor_ckpt(ckpt)
            # match = re.search(r'(\d+)(?=\D*$)', ckpt)
            # if match:
            #     step = int(match.group(1))
            
            # Note
            i += 1
            agents.append((agent, i, elbow_rand, shoulder_rand))
                
            print('Num agents', len(agents))


    garment_ids = [1, 2]
    motion_ids = [0, 1, 2, 4, 5, 6, 7, 8]
    # garment_ids = [1]
    # motion_ids = [0]

    # pose_ids = [i for i in range(28)]

    # Note
    # pairs = [(x, 0, y) for x in garment_ids for y in pose_ids]
    pairs = [(x, y, -1) for x in garment_ids for y in motion_ids]

    # Note
    print('Num pairs', int(len(pairs)))
        
    for agent, step, elbow_rand, shoulder_rand in agents:
        episode = int(step)
        print("Eval begin step = {}".format(step))
        L.log('eval-number', episode, step)

        # Note
        parallel_evaluator = ParallelEvaluator(num_workers=int(len(pairs)))
        dressed_ratios = parallel_evaluator.evaluate_agents(agent, step, elbow_rand, shoulder_rand, pairs, args)
        parallel_evaluator.close()

        L.log('eval/mean_upperarm_ratio', dressed_ratios, step)
        L.dump(step)

        if args.use_wandb: wandb.log({'mean_upper_arm_ratios': dressed_ratios, 'step': step})


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
        
    exp_prefix =  '2025-0302-pybullet-eval-ckpt'
    load_variant_path = '/home/alexis/assistive-gym-film/assistive_gym/envs/variant_fcvp.json'
    loaded_vg = create_vg_from_json(load_variant_path)
    print("Loaded configs from ", load_variant_path)
    vg = loaded_vg
    print(vg)
    print('Number of configurations: ', len(vg.variants()))
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    exp_count = 0
    timestamp = now.strftime('%m_%d_%H_%M_%S')
    exp_name = "test_fcvp"
    # exp_name = "iql-training-p1-reward_model1-only_eval-t6"

    print(exp_name)
    exp_name = "{}-{}-{:03}".format(exp_name, timestamp, exp_count)
    log_dir = '/scratch/alexis/data/' + exp_prefix + "/" + exp_name

    run_task(vg, log_dir=log_dir, exp_name=exp_name)
    #eval()