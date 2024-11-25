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


# import cProfile, pstats, io

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
    if vv['resume_from_ckpt']:
        log_dir = vv['resume_from_exp_path']
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
def evaluate(env, agent, video_dir, L, step, args, get_step_func=None):
    factor = args.save_gif_factor
    env._wrapped_env.eval_flag = True
    if step >= args.saving_gif_start and (step // args.eval_freq) % factor == 0:
        env._wrapped_env.show_reward = True
        imsize = 720

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = env._wrapped_env.garments[0] + '_'
        all_traj_returns_garments_poses = defaultdict(lambda: defaultdict(list))
        # {region_num_0: {garment0: [pose0, pose1, ...], garment1: [...], ....., garment4: [...]}, ...., region_num_27: ...}
        all_upperarm_ratio_garments_poses = defaultdict(lambda: defaultdict(list))

        args.all_eval_poses = args.eval_poses
        # breakpoint()
        gif_saved = False
        if step >= args.saving_gif_start and (step // args.eval_freq) % factor == 0 and not gif_saved:
            plt.figure()

        cnt = 0
        garment_ids = [1, 2]
        motion_ids = [0, 2, 4, 5, 6, 7]
        
        for garment_id in garment_ids:

            env.cur_garment_idx = garment_id
            env.set_garment(env.cur_garment_idx)

            for motion_id in motion_ids:
                
                obs = env.reset(garment_id=garment_id, motion_id=motion_id)
                done = False
                episode_reward = 0
                ep_info = []
                rewards = []

                # if step >= args.saving_gif_start and (step // args.eval_freq) % factor == 0 and not gif_saved:
                #     initial_images = env.get_image(imsize, imsize, both_camera_angle=True)
                #     frames, frames_2 = [initial_images[0]], [initial_images[1]]
                #     pc_images, reward_images = [], []
                #     all_images = []
                
                t = 0
                while not done:
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

                    obs, reward, done, info = env.step(step_action)
                    episode_reward += reward
                    ep_info.append(info)
                    rewards.append(reward)

                    # if step >= args.saving_gif_start and (step // args.eval_freq) % factor == 0 and not gif_saved:
                    #     image = env.get_image(imsize, imsize, both_camera_angle=False)
                    #     frames.append(image)
                    #     pc_images.append(cv2.resize(env.reward_image, (imsize, imsize)))

                    #     image_from_plot = get_train_plot_image(rewards, imsize, ep_info, info)
                    #     reward_images.append(image_from_plot)
                    #     all_image = np.concatenate([frames[-1], pc_images[-1]], axis=1)
                    #     all_image = np.concatenate([all_image, reward_images[-1]], axis=1)
                    #     all_images.append(all_image)
                    t += 1


                # if step >= args.saving_gif_start and (step // args.eval_freq) % factor == 0 and not gif_saved and len(all_images) > 0:
                #     save_numpy_as_gif(np.array(all_images), os.path.join(video_dir, '{}-{}-{}-{}.gif'.format(step, garments[garment_id], region_id, pose_id)))

                all_traj_returns_garments_poses[garment_id][motion_id].append(episode_reward)
                all_upperarm_ratio_garments_poses[garment_id][motion_id].append(info['upperarm_ratio'])
                L.log('eval/garmemt{}_motion{}_episode_reward'.format(garment_id, motion_id), episode_reward, step)
                if args.use_wandb: 
                    wandb.log({'garmemt{}_motion{}_episode_reward'.format(garment_id, motion_id): episode_reward, 'step': step})
                    wandb.log({'garmemt{}_motion{}_episode_dressed_ratio'.format(garment_id, motion_id): info['upperarm_ratio'], 'step': step})

                for key, val in get_info_stats([ep_info]).items():
                    L.log('eval/garmemt{}_motion{}_info_'.format(garment_id, motion_id) + key, val, step)

                cnt += 1

        gif_saved = False
                    
        # if step >= args.saving_gif_start and (step // args.eval_freq) % factor == 0:
        #     plt.close("all")

        eval_time = time.time() - start_time
        L.log('eval/' + prefix + 'wallclock_time', np.mean(eval_time), step)

        garment_reward_means = []
        garment_upperarm_ratio_means = []
        # log reward for every motion, averaged across garment
        for motion_id in motion_ids:
            prefix = motion_id + '_'
            all_traj_returns_all_poses = np.array([
                all_traj_returns_garments_poses[garment_id][garment_id] 
                for garment_id in garment_ids 
                if not (
                    isinstance(all_traj_returns_garments_poses[garment_id][motion_id], list) and
                    len(all_traj_returns_garments_poses[garment_id][motion_id]) == 0
                )
            ])
            all_upperarm_ratio_all_poses = np.array([
                all_upperarm_ratio_garments_poses[garment_id][motion_id] 
                for garment_id in garment_ids 
                if not (
                    isinstance(all_traj_returns_garments_poses[garment_id][motion_id], list) and
                    len(all_traj_returns_garments_poses[garment_id][motion_id]) == 0
                )
            ])

            if (len(all_traj_returns_all_poses) == 0): continue

            mean_ep_reward, best_ep_reward = np.mean(all_traj_returns_all_poses), np.max(all_traj_returns_all_poses)
            mean_upperarm_ratio, best_upperarm_ratio = np.mean(all_upperarm_ratio_all_poses), np.max(all_upperarm_ratio_all_poses)
            L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
            L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)
            L.log('eval/' + prefix + 'mean_upperarm_ratio', mean_upperarm_ratio, step)
            L.log('eval/' + prefix + 'best_upperarm_ratio', best_upperarm_ratio, step)

            if args.use_wandb: 
                wandb.log({'motion{}_mean_ep_reward'.format(motion_id): mean_ep_reward, 'step': step})
                wandb.log({'motion{}_best_ep_reward'.format(motion_id): best_ep_reward, 'step': step})
                wandb.log({'motion{}_mean_upper_ratio'.format(motion_id): mean_upperarm_ratio, 'step': step})
                wandb.log({'motion{}_best_upper_ratio'.format(motion_id): best_upperarm_ratio, 'step': step})


            garment_reward_means.append(mean_ep_reward)
            garment_upperarm_ratio_means.append(mean_upperarm_ratio)
            
        mean_ep_reward = np.mean(garment_reward_means)
        mean_upperarm_ratio = np.mean(garment_upperarm_ratio_means)
        L.log('eval/mean_episode_reward', mean_ep_reward, step)
        L.log('eval/mean_upperarm_ratio', mean_upperarm_ratio, step)

        if args.use_wandb: 
            wandb.log({'mean_episode_reward': mean_ep_reward, 'step': step})
            wandb.log({'mean_upperarm_ratio': mean_upperarm_ratio, 'step': step})


        return mean_ep_reward, mean_upperarm_ratio
    mean_ep_reward, mean_upperarm_ratio = run_eval_loop(sample_stochastically=False)
    L.dump(step)
    # if step >= args.saving_gif_start and (step // args.eval_freq) % factor == 0:
    #     env._wrapped_env.show_reward = False

    return mean_ep_reward, mean_upperarm_ratio

def make_agent(obs_shape, action_shape, args, device, agent="IQL"):
    # create the sac trainning agent

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
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs

    env = DressingSawyerHumanEnv(policy=args.policy, horizon=args.horizon, camera_pos=args.camera_pos, rand=args.rand, render=args.render)

    # make directory for logging
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    dataset_dir = args.dataset_dir
    args.work_dir = '/scratch/alexis/data/'
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))
    L = Logger(args.work_dir, use_tb=args.save_tb, chester_logger=logger)

    # create replay buffer
    device = torch.device('cuda:{}'.format(args.cuda_idx) if torch.cuda.is_available() else 'cpu')
    action_shape = (6,)
    obs_shape = (30000,)
    replay_buffers = []
    rb_limit = args.replay_buffer_capacity // args.replay_buffer_num
    args.encoder_type = 'pointcloud_flow'
    args.pc_feature_dim = 2
    buffer = PointCloudReplayBuffer(
        args, action_shape, rb_limit, args.batch_size, device, td=args.__dict__.get("td", False), n_step=args.__dict__.get("n_step", 1),reward_relabel=args.reward_relabel
    )
    buffer.load2(dataset_dir)
    reward_model1 = RewardModelVLM(obs_shape, action_shape, args, use_action=args.reward_model_use_action)
    reward_model1.load(args.reward_model1_dir, args.reward_model1_step)
    reward_model1.eval()
    
    reward_model2 = RewardModelVLM(obs_shape, action_shape, args, use_action=args.reward_model_use_action)
    reward_model2.load(args.reward_model2_dir, args.reward_model2_step)
    reward_model2.eval()

    if args.reward_relabel:
        with torch.no_grad():
            buffer.relabel_rewards(reward_model1, reward_model2)
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
        agent.load(args.resume_from_path_actor, args.resume_from_path_critic, load_optimizer=args.resume_from_ckpt)
        ckpt  = torch.load(osp.join(args.resume_from_path_critic))

        
    for step in tqdm(range(args.num_train_steps)):
        # evaluate agent periodically
        if step%args.eval_freq == 0 and step > 0:
            episode = int(step/args.eval_freq )
            print("Eval begin step = {}".format(step))
            L.log('eval-number', episode, step)
            # print("Eval is happening")
            avg_traj_returns_unseen, avg_upper_arm_ratios_unseen = evaluate(env, agent, video_dir, L, step, args)
            if avg_upper_arm_ratios_unseen > best_avg_upper_arm_ratios_unseen:
                best_avg_upper_arm_ratios_unseen = avg_upper_arm_ratios_unseen
                if args.use_wandb: wandb.log({'avg_upper_arm_ratios': avg_upper_arm_ratios_unseen, 'step': step})

                agent.save(model_dir, step, episode, is_best_test=True, best_avg_return_test=round(best_avg_upper_arm_ratios_unseen, 5))
           
            if args.save_model:
                agent.save(model_dir, step, episode)

            if args.save_buffer:
                for i in range(args.replay_buffer_num):
                    replay_buffers[i].save(buffer_dir, i)

            if args.evaluate_only:
                exit()

        # run training update
        agent.update(replay_buffers, L, step, pose_id=0)


if __name__ == "__main__":
        
    exp_prefix =  '2024-1124-pybullet-finetuning-simple'
    load_variant_path = '/home/alexis/assistive-gym-fem/assistive_gym/envs/variant_real_world_final.json'
    loaded_vg = create_vg_from_json(load_variant_path)
    print("Loaded configs from ", load_variant_path)
    vg = loaded_vg
    print(vg)
    print('Number of configurations: ', len(vg.variants()))
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    exp_count = 0
    timestamp = now.strftime('%m_%d_%H_%M_%S')
    exp_name = "iql-training-p2-reward_model-7030-step140"
    print(exp_name)
    exp_name = "{}-{}-{:03}".format(exp_name, timestamp, exp_count)
    log_dir = '/scratch/alexis/data/' + exp_prefix + "/" + exp_name

    run_task(vg, log_dir=log_dir, exp_name=exp_name)
    #eval()