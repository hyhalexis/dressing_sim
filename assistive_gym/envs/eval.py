import numpy as np
import torch
import os
import time
import json  
import copy
import multiprocessing as mp
import os.path as osp
import argparse
import matplotlib.pyplot as plt
from chester import logger


import utils

from visualization import save_numpy_as_gif
from SAC_AWAC import SAC_AWACAgent


import cv2, pickle
import cProfile, pstats, io
import matplotlib
from dressing_envs import DressingSawyerHumanEnv
# matplotlib.use('agg')


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
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir, exp_name=exp_name, format_strs=['csv'])
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    load_vv = json.load(open(vv["variant_path"], 'r'))
    # print(load_vv)
    load_vv.update(**vv) 
    main(vv_to_args(load_vv))   


def evaluate(env, agent, video_dir, traj_dir, step, args):
    all_ep_rewards = []
    imsize = 720

    def run_eval_loop():
        start_time = time.time()
        infos = []
        final_rewards = []
        all_whole_arm_ratios = []
        all_upper_arm_ratios = []
        traj_rewards = []

        # Start
        garment_idx = 2
        traj_obses = []
        obs = env.reset()
        traj_obses.append(obs)
        done = False
        episode_reward = 0
        ep_info = []
        whole_arm_ratios, upper_arm_ratios = [], []
        actions_list = []

        # initial_images = env.get_image(imsize, imsize, both_camera_angle=True)
        # frames, frames_2 = [initial_images[0]], [initial_images[1]]

        pc_images = []
        # reward_images = []
        # obs_images = []
        # all_images = []
        rewards = []
        distance_tool_to_human = []
        traj_dataset = []
        
        # pos_grad_images = []
        # feature_grad_images = []

        # with open('traj_data/actions26.pkl', 'rb') as f:
        #     actions = pickle.load(f)
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

            print('action', step_action)
            actions_list.append(step_action)
            # step_action = actions[t]
            new_obs, reward, done, info = env.step(step_action)

            step_data = {
                'obs': obs,
                'action': step_action,
                'new_obs': new_obs,
                'reward': reward,
                'info': info,
                'img': env.step_img,
                'gripper_pos': env.step_gripper_pos,
                'line_points': env.step_line_pts
            }

            traj_dataset.append(step_data)
            
            # with open(os.path.join(traj_dir, 'transition_{}.pkl'.format(t)), 'wb') as f:
            #     pickle.dump(step_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            obs = new_obs
            if obs is None: # collision avoidance-skip action
                continue
            traj_obses.append(obs)
            whole_arm_ratios.append(info['whole_arm_ratio'])
            upper_arm_ratios.append(info['upperarm_ratio'])

            episode_reward += reward
            ep_info.append(info)
            rewards.append(reward)

            # images = env.get_image(imsize, imsize, both_camera_angle=True) # XXX
            # frames.append(images[0]), frames_2.append(images[1])

            pc_images.append(cv2.resize(env.reward_image, (imsize, imsize)))
            # all_image = np.concatenate([frames[-1], frames_2[-1], pc_images[-1]], axis=1)
            
            # plt.close("all")
            # all_images.append(all_image)
            t += 1

        # with open(os.path.join(video_dir, '{}.pkl'.format(env.garment)), 'wb') as f:
        #     pickle.dump(traj_obses, f)

        with open(os.path.join(traj_dir, 'p{}_motion{}_{}_{}_{}.pkl'.format(env.policy, env.motion_id, env.camera_pos, env.garment, np.max(upper_arm_ratios))), 'wb') as f:
            pickle.dump(traj_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


        # with open(os.path.join(traj_dir, 'actions26.pkl'), 'wb') as f:
        #     print('actions saved')
        #     pickle.dump(actions_list, f, protocol=pickle.HIGHEST_PROTOCOL)

        # if False:
        save_numpy_as_gif(np.array(pc_images), os.path.join(video_dir, '{}.gif'.format(env.garment)))

        infos.append(ep_info)
        traj_rewards.append(rewards)
        plt.plot(rewards)
        plt.savefig(os.path.join(video_dir, 'rewards.png'))
        plt.close('all')

        plt.plot(whole_arm_ratios)
        plt.savefig(os.path.join(video_dir, 'whole_arm_ratios.png'))
        plt.close('all')

        plt.plot(upper_arm_ratios)
        plt.savefig(os.path.join(video_dir, 'upper_arm_ratios.png'))
        plt.close('all')


        print("whole arm ratio {} upper arm ratio {}".format(info['whole_arm_ratio'], info['upperarm_ratio']))
        all_whole_arm_ratios.append(np.max(whole_arm_ratios))
        all_upper_arm_ratios.append(np.max(upper_arm_ratios))
        all_ep_rewards.append(episode_reward)
        
        plt.close("all")
        for rewards in traj_rewards:
            plt.plot(range(len(rewards)), rewards)
        plt.savefig(os.path.join(video_dir, '%d.png' % step))
        plt.close("all")

        eval_time = time.time() - start_time
        mean_ep_reward = np.mean(all_ep_rewards)
        std_ep_reward = np.std(all_ep_rewards) if len(all_ep_rewards) > 1 else 0.0
        best_ep_reward = np.max(all_ep_rewards)
        return all_whole_arm_ratios, all_upper_arm_ratios
        
    all_whole_arm_ratios, all_upper_arm_ratios = run_eval_loop()
    return all_whole_arm_ratios, all_upper_arm_ratios

def make_agent(obs_shape, action_shape, args, device):
    # agent_class = SAC_AWACAgent
    # return agent_class(
    #     args=args,
    #     obs_shape=obs_shape,
    #     action_shape=action_shape,
    #     device=device,
    #     actor_load_name=args.actor_load_name,
    # )
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
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    env = DressingSawyerHumanEnv(policy=args.policy, motion=args.motion_id, garment=args.garment_id, horizon=args.horizon, camera_pos=args.camera_pos, render=args.render)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)

    args.work_dir = logger.get_dir()

    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))

    traj_dir = 'traj_data'
    if not os.path.exists(traj_dir):
        os.mkdir(traj_dir)

    device = torch.device('cuda:{}'.format(args.cuda_idx) if torch.cuda.is_available() else 'cpu')

    action_shape = (6,)
    obs_shape = (30000,)

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    all_whole_arm_ratios, all_upper_arm_ratios = evaluate(env, agent, video_dir, traj_dir, 0, args)
    np.save(osp.join(args.work_dir, 'all_upper_arm_ratios.npy'), all_upper_arm_ratios)
    np.save(osp.join(args.work_dir, 'all_whole_arm_ratios.npy'), all_whole_arm_ratios)
    
    plt.plot(all_upper_arm_ratios)
    plt.xlabel("joint movement")
    plt.ylabel("upperarm ratio")
    plt.savefig(osp.join(args.work_dir, 'all_upper_arm_ratios.png'))
    plt.close("all")
    
    plt.plot(all_whole_arm_ratios)
    plt.xlabel("joint movement")
    plt.ylabel("whole arm ratio")
    plt.savefig(osp.join(args.work_dir, 'all_whole_arm_ratios.png'))
    plt.close("all")
    
    print("all_whole_arm_ratios: ", all_whole_arm_ratios)
    print("mean all_traj_return: ", np.mean(all_whole_arm_ratios))
    print("all_upper_arm_ratios", all_upper_arm_ratios)
    print("mean all_upper_arm_ratios", np.mean(all_upper_arm_ratios))

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    
    main()

