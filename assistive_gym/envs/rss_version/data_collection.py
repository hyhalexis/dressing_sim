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

from chester import logger
import utils
from rss_version.default_config import DEFAULT_CONFIG
from logger import Logger
from dressing_envs import DressingSawyerHumanEnv
from rss_version.SAC_AWAC import SAC_AWACAgent

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

    def evaluate_agents(self, pairs, args):
        task_args = [
            (agent, step, garment_id, motion_id, pose_id, args)
            for agent, step, garment_id, motion_id, pose_id in pairs
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
    agent, step, garment_id, motion_id, pose_id, args = arg

    def run_eval_loop():
        start_time = time.time()
        prefix = '_'
        all_traj_returns_garments_poses = defaultdict(lambda: defaultdict(list))
        # {region_num_0: {garment0: [pose0, pose1, ...], garment1: [...], ....., garment4: [...]}, ...., region_num_27: ...}
        all_upperarm_ratio_garments_poses = defaultdict(lambda: defaultdict(list))

        cnt = 0
        env = DressingSawyerHumanEnv(policy=3, horizon=args.horizon, camera_pos=args.camera_pos, occlusion=args.occlusion, use_force=args.use_force, render=args.render, gif_path=args.gif_path)

        obs = env.reset(garment_id=garment_id, motion_id=motion_id, pose_id=pose_id, step_idx = step)
        done = False
        episode_reward = 0
        ep_info = []
        rewards = []
        traj_dataset = []

        t = 0
        while not done:
            print('------Iteration', t)

            with utils.eval_mode(agent):
                action = agent.select_action(obs)
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

            # if t > 30:
            #     rng = np.random.default_rng()
            #     rand_action = rng.uniform(-0.025, 0.025, size=3)
            #     step_action[:3] += rand_action

            new_obs, reward, done, info = env.step(step_action)
            
            step_data = {
                'obs': obs,
                'action': action,
                'new_obs': new_obs,
                'reward': reward,
                'done': done,
                'info': info,
                'img': env.step_img,
                'gripper_pos': env.step_gripper_pos,
                'line_points': env.step_line_pts,
                'robot_force': env.robot_force_on_human,
                'cloth_force': env.cloth_force_sum,
                'total_force': env.total_force_on_human
            }
            
            traj_dataset.append(step_data)

            obs = new_obs
            episode_reward += reward
            ep_info.append(info)
            rewards.append(reward)
            t += 1
        
        with open(os.path.join(args.traj_dir, 'p{}_motion{}_{}_{}_{}_{}_{}_{}.pkl'.format(3, env.motion_id, env.camera_pos, env.garment, int(env.shoulder_rand), int(env.elbow_rand), info['whole_arm_ratio'], info['upperarm_ratio'])), 'wb') as f:
            pickle.dump(traj_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

        env.disconnect()
        return info['upperarm_ratio']

        
    upperarm_ratio = run_eval_loop()
    # L.log('eval/garment{}_motion{}_upperarm_ratio'.format(garment_id, motion_id), upperarm_ratio, step)
    # L.dump(step)
    return upperarm_ratio

def make_agent(obs_shape, action_shape, args, device):

    agent_class = SAC_AWACAgent
    return agent_class(
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
        agent=args.agent,
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
    print('Occlusion?', args.occlusion)

    # make directory for logging
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)

    # create replay buffer
    # device = torch.device('cuda:{}'.format(args.cuda_idx) if torch.cuda.is_available() else 'cpu')
    device = 'cuda:0'
    action_shape = (6,)
    obs_shape = (30000,)
    
    # folder_path = "/scratch/alexis/data/2024-1205-pybullet-finetuning/iql-training-p1-reward_model1-only-12_05_07_16_53-000/model"
    folder_path = "/scratch/alexis/data/2024-1205-pybullet-finetuning/iql-training-p2-reward_model1-only-12_05_05_50_38-000/model"
    
     # Note
    dir = '/scratch/alexis/data/traj_data_with_force_one-hot'

    args.traj_dir = '{}/trajs'.format(dir)
    if not os.path.exists(args.traj_dir):
        os.makedirs(args.traj_dir)

    # dir = os.path.dirname(folder_path)

    L = Logger(dir, use_tb=args.save_tb, chester_logger=logger)

    gif_path = '{}/gifs'.format(dir)
    os.makedirs(gif_path, exist_ok=True)
    print('Saving gifs at: ', gif_path)
    args.gif_path = gif_path
    import re
        # Regular expression pattern to match 'actor_xxx.pt' where xxx is a number
    filename_pattern = re.compile(r"actor_(\d+)\.pt$")

    # # List to store the matching filenames
    # lst = [30000, 40000, 50000, 60000, 110000, 140000, 150000, 160000, 170000, 190000, 210000, 220000]

    agents = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename_pattern.match(filename) and int(filename_pattern.match(filename).group(1)) > 70000:  # If the filename matches the pattern
            agents.append(filename)
    agents.sort(key=lambda x: int(filename_pattern.match(x).group(1)))

    agents_ckpts = [
                    # "/home/alexis/assistive-gym-film/assistive_gym/envs/ckpt/actor_1900106.pt",
                    # "/home/alexis/assistive-gym-film/assistive_gym/envs/ckpt/actor_best_test_600023_0.65914.pt",
                    "/scratch/alexis/data/1218_flex_one-hot/model/actor_best_test_600057_0.73525.pt"
    ]           

    # agents_ckpts = ["/home/alexis/assistive-gym-film/assistive_gym/envs/ckpt_rss/actor_160111.pt"]
         
    # Note
    agents = []
    import re
    i = 0
    for ckpt in agents_ckpts:
        agent = make_agent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            args=args,
            device=device
        )
        
        agent.load_actor_ckpt(ckpt)
        # match = re.search(r'(\d+)(?=\D*$)', ckpt)
        # if match:
        #     step = int(match.group(1))
        
        # Note
        agents.append((agent, i))
        i += 1
    print('Num agents', len(agents))


    garment_ids = [1, 2]
    # Note
    motion_ids = [0, 2, 4, 5, 6, 7]
    # pose_ids = [i for i in range(28)]

    # Note
    # pairs = [(x, 0, y) for x in garment_ids for y in pose_ids]
    pairs = [(a, s, x, y, -1) for a, s in agents for x in garment_ids for y in motion_ids]

    # Note
    print('Num pairs', int(len(pairs)))

    for _ in range(20):
        parallel_evaluator = ParallelEvaluator(num_workers=int(len(pairs)))
        dressed_ratios = parallel_evaluator.evaluate_agents(pairs, args)
        parallel_evaluator.close()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
        
    exp_prefix =  '2024-1224-pybullet-eval-ckpt'
    load_variant_path = '/home/alexis/assistive-gym-film/assistive_gym/envs/rss_version/variant_rss.json'
    loaded_vg = create_vg_from_json(load_variant_path)
    print("Loaded configs from ", load_variant_path)
    vg = loaded_vg
    print(vg)
    print('Number of configurations: ', len(vg.variants()))
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    exp_count = 0
    timestamp = now.strftime('%m_%d_%H_%M_%S')
    exp_name = "data_collection"
    # exp_name = "iql-training-p1-reward_model1-only_eval-t6"

    print(exp_name)
    exp_name = "{}-{}-{:03}".format(exp_name, timestamp, exp_count)
    log_dir = '/scratch/alexis/data/' + exp_prefix + "/" + exp_name

    run_task(vg, log_dir=log_dir, exp_name=exp_name)
    #eval()