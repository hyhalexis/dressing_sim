from collections import defaultdict
import matplotlib
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
from bc_pointcloud import BCAgent
from visualization import save_numpy_as_gif
from dressing_envs import DressingSawyerHumanEnv

# import cProfile, pstats, io

### this function transforms the policy outputted action to be the action that is executable in the environment
# e.g., for dense transformation policy, this extracts the action corresponding to the gripper point, which is the last action in the action array
def get_step_action(action, args, obs=None, scale=None):
    if not args.dual_arm:
        if 'flow' in args.encoder_type:
            step_action = action.reshape(-1, 6)[-1].flatten()
        else:
            step_action = action.copy()

        if 'clip_rotation_mode' in args.__dict__: 
            if args.clip_rotation_mode == 'to_y':
                step_action[[3, 5]] = 0
            elif args.clip_rotation_mode == 'to_yz':
                step_action[3] = 0
        else:
            if args.clip_rotation_to_y:
                step_action[[3, 5]] = 0
    else:
        if 'flow' in args.encoder_type:
            step_action = action.reshape(-1, 6)[-2:].flatten()
        else:
            step_action = action.copy()

        if args.clip_rotation_mode == 'to_y':
            step_action[[3, 5, 3+6, 5+6]] = 0
        elif args.clip_rotation_mode == 'to_yz':
            step_action[[3, 3+6]] = 0

    return step_action

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
        wandb.init(project="dressing-flow", name=exp_name, entity="conan-iitkgp", config=updated_vv, settings=wandb.Settings(start_method='fork'))

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

    garments = args.garments
    garments_num = len(garments)
    eval_regions = args.eval_regions
    per_region_pose_num = args.__dict__.get('per_region_pose_num', 50)
    per_region_config_num = per_region_pose_num * garments_num 

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

        assert args.num_of_garments_to_eval <= garments_num
        cnt = 0
        print("here")
        for region_id in eval_regions:
            if args.faster_eval:
                if region_id%2 == 0:
                    # continue # only eval odd regions
                    gif_saved = True # only save gif on odd regions
                garment_ids = np.random.choice(range(garments_num), size=args.num_of_garments_to_eval, replace=False) # random subset
            else:
                garment_ids = range(garments_num)
            
            for garment_id in garment_ids:

                env.cur_garment_idx = garment_id
                env.set_garment(env.cur_garment_idx)
                
                for pose_id in args.all_eval_poses:

                    config_idx = region_id * per_region_config_num + pose_id * garments_num + garment_id
                    print("Eval: {} th episode - config id - {} region id - {} pose id - {} garment_id - {}".format(cnt, config_idx, region_id, pose_id, garment_id))
                    
                    obs = env.reset(config_id=config_idx)
                    done = False
                    episode_reward = 0
                    ep_info = []
                    rewards = []

                    if step >= args.saving_gif_start and (step // args.eval_freq) % factor == 0 and not gif_saved:
                        initial_images = env.get_image(imsize, imsize, both_camera_angle=True)
                        frames, frames_2 = [initial_images[0]], [initial_images[1]]
                        pc_images, reward_images = [], []
                        all_images = []
                    
                    t = 0
                    while not done:
                        if args.agent == 'curl_sac':
                            with utils.eval_mode(agent):
                                if sample_stochastically:
                                    action = agent.sample_action(obs)
                                else:
                                    action = agent.select_action(obs)
                        else:
                            action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
                            action = action.cpu().numpy()

                        if get_step_func is None:
                            step_action = get_step_action(action, args, obs)
                        else:
                            step_action = get_step_func(action, args, obs)

                        obs, reward, done, info = env.step(step_action)
                        episode_reward += reward
                        ep_info.append(info)
                        rewards.append(reward)

                        if not info['simulator_error']:
                            if step >= args.saving_gif_start and (step // args.eval_freq) % factor == 0 and not gif_saved:
                                image = env.get_image(imsize, imsize, both_camera_angle=False)
                                frames.append(image)
                                pc_images.append(cv2.resize(env.reward_image, (imsize, imsize)))

                                image_from_plot = get_train_plot_image(rewards, imsize, ep_info, info)
                                reward_images.append(image_from_plot)
                                all_image = np.concatenate([frames[-1], pc_images[-1]], axis=1)
                                all_image = np.concatenate([all_image, reward_images[-1]], axis=1)
                                all_images.append(all_image)
                            t += 1
                        else:
                            done = True

                    if step >= args.saving_gif_start and (step // args.eval_freq) % factor == 0 and not gif_saved and len(all_images) > 0:
                        save_numpy_as_gif(np.array(all_images), os.path.join(video_dir, '{}-{}-{}-{}.gif'.format(step, garments[garment_id], region_id, pose_id)))
                        if args.faster_eval:
                            gif_saved = True

                    all_traj_returns_garments_poses[region_id][garment_id].append(episode_reward)
                    all_upperarm_ratio_garments_poses[region_id][garment_id].append(info['upperarm_ratio'])
                    L.log('eval/config_{}_episode_reward'.format(config_idx), episode_reward, step)
                    for key, val in get_info_stats([ep_info]).items():
                        L.log('eval/config_{}_info_'.format(config_idx) + key, val, step)

                    cnt += 1

                    if args.faster_eval:
                        break # evaluate on only the first pose
            gif_saved = False
                    
        if step >= args.saving_gif_start and (step // args.eval_freq) % factor == 0:
            plt.close("all")

        eval_time = time.time() - start_time
        L.log('eval/' + prefix + 'wallclock_time', np.mean(eval_time), step)

        garment_reward_means = []
        garment_upperarm_ratio_means = []
        # log reward for every garment, averaged across pose regions
        for garment_id in range(garments_num):
            prefix = garments[garment_id] + '_'
            all_traj_returns_all_poses = np.array([
                all_traj_returns_garments_poses[region_id][garment_id] 
                for region_id in eval_regions 
                if not (
                    isinstance(all_traj_returns_garments_poses[region_id][garment_id], list) and
                    len(all_traj_returns_garments_poses[region_id][garment_id]) == 0
                )
            ])
            all_upperarm_ratio_all_poses = np.array([
                all_upperarm_ratio_garments_poses[region_id][garment_id] 
                for region_id in eval_regions 
                if not (
                    isinstance(all_traj_returns_garments_poses[region_id][garment_id], list) and
                    len(all_traj_returns_garments_poses[region_id][garment_id]) == 0
                )
            ])

            if (len(all_traj_returns_all_poses) == 0): continue

            mean_ep_reward, best_ep_reward = np.mean(all_traj_returns_all_poses), np.max(all_traj_returns_all_poses)
            mean_upperarm_ratio, best_upperarm_ratio = np.mean(all_upperarm_ratio_all_poses), np.max(all_upperarm_ratio_all_poses)
            L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
            L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)
            L.log('eval/' + prefix + 'mean_upperarm_ratio', mean_upperarm_ratio, step)
            L.log('eval/' + prefix + 'best_upperarm_ratio', best_upperarm_ratio, step)

            garment_reward_means.append(mean_ep_reward)
            garment_upperarm_ratio_means.append(mean_upperarm_ratio)
            
        # log reward for every pose region, averaged across garments
        for region_id in eval_regions:
            # if args.faster_eval and region_id%2==0: continue
            prefix = 'region_' + str(region_id) + '_'
            all_traj_returns_all_garments = np.array([
                all_traj_returns_garments_poses[region_id][garment_id]
                for garment_id in range(garments_num) 
                if not (
                    isinstance(all_traj_returns_garments_poses[region_id][garment_id], list) and
                    len(all_traj_returns_garments_poses[region_id][garment_id]) == 0
                )
            ])
            all_upperarm_ratio_all_garments = np.array([
                all_upperarm_ratio_garments_poses[region_id][garment_id]
                for garment_id in range(garments_num)
                if not (
                    isinstance(all_traj_returns_garments_poses[region_id][garment_id], list) and
                    len(all_traj_returns_garments_poses[region_id][garment_id]) == 0
                )
            ])
            mean_ep_reward, best_ep_reward = np.mean(all_traj_returns_all_garments), np.max(all_traj_returns_all_garments)
            mean_upperarm_ratio, best_upperarm_ratio = np.mean(all_upperarm_ratio_all_garments), np.max(all_upperarm_ratio_all_garments)
            L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
            L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)
            L.log('eval/' + prefix + 'mean_upperarm_ratio', mean_upperarm_ratio, step)
            L.log('eval/' + prefix + 'best_upperarm_ratio', best_upperarm_ratio, step)
        
        mean_ep_reward = np.mean(garment_reward_means)
        mean_upperarm_ratio = np.mean(garment_upperarm_ratio_means)
        L.log('eval/mean_episode_reward', mean_ep_reward, step)
        L.log('eval/mean_upperarm_ratio', mean_upperarm_ratio, step)

        return mean_ep_reward, mean_upperarm_ratio
    mean_ep_reward, mean_upperarm_ratio = run_eval_loop(sample_stochastically=False)
    L.dump(step)
    env._wrapped_env.eval_flag = False
    if step >= args.saving_gif_start and (step // args.eval_freq) % factor == 0:
        env._wrapped_env.show_reward = False

    return mean_ep_reward, mean_upperarm_ratio

def make_agent(obs_shape, action_shape, args, device, agent="IQL"):
    # create the sac trainning agent
    if agent=="IQL":
        agent_class = IQLAgent
    else:
        agent_class = BCAgent
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

    # build environment
    env = DressingSawyerHumanEnv(render=args.render)


    # make directory for logging
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    dataset_dir = args.dataset_dir
    args.work_dir = logger.get_dir()
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))
    L = Logger(args.work_dir, use_tb=args.save_tb, chester_logger=logger)

    # create replay buffer
    device = torch.device('cuda:{}'.format(args.cuda_idx) if torch.cuda.is_available() else 'cpu')
    action_shape = [6]
    obs_shape = env.observation_space.shape
    replay_buffers = []
    rb_limit = args.replay_buffer_capacity // args.replay_buffer_num
    args.encoder_type = 'pointcloud_flow'
    buffer = PointCloudReplayBuffer(
        args, action_shape, rb_limit, args.batch_size, device, td=args.__dict__.get("td", False), n_step=args.__dict__.get("n_step", 1),reward_relabel=args.reward_relabel
    )
    buffer.load(dataset_dir, rb_idx=0, fix_load_data=False)
    reward_model = RewardModelVLM(obs_shape, action_shape, args, use_action=args.reward_model_use_action)
    reward_model.load(args.reward_model_dir, args.reward_model_step)
    reward_model.eval()
    if args.reward_relabel:
        with torch.no_grad():
            buffer.relabel_rewards(reward_model)
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
    garments = args.garments
    garments_num = len(garments)

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
        
    exp_prefix =  '2024-0908-real-world-dressing-experiments'
    load_variant_path = "/home/sreyas/Desktop/Projects/softagent_rpad/dressing_motion/curl/variant_real_world_final.json"
    loaded_vg = create_vg_from_json(load_variant_path)
    print("Loaded configs from ", load_variant_path)
    vg = loaded_vg
    print(vg)
    print('Number of configurations: ', len(vg.variants()))
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    exp_count = 0
    timestamp = now.strftime('%m_%d_%H_%M_%S')
    exp_name = "iql-training-real-world-iql-vlm-reward-gpt4-reward_model-step-300"
    print(exp_name)
    exp_name = "{}-{}-{:03}".format(exp_name, timestamp, exp_count)
    log_dir = "data" + "/local/" + exp_prefix + "/" + exp_name

    run_task(vg, log_dir=log_dir, exp_name=exp_name)
    #eval()