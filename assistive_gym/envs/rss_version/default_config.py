import argparse

DEFAULT_CONFIG = {
    # environment
    'domain_name': 'cartpole',
    'task_name': 'swingup',
    'pre_transform_image_size': 128,
    'image_size': 100,
    'action_repeat': 1,
    # replay buffer
    'replay_buffer_capacity': 100000,
    # train
    'agent': 'curl_sac',
    'init_steps': 1000,
    'num_train_steps': 2000000,
    'batch_size': 128,  # This is 256 for other sac but experiments show that they don't matter much for curl_sac
    'hidden_dim': 1024,
    # eval
    'eval_freq': 10000,
    'num_eval_episodes': 10,
    # critic
    'critic_lr': 1e-3,
    'critic_beta': 0.9,
    'critic_tau': 0.01,  # try 0.05 or 0.1
    'critic_target_update_freq': 2,  # try to change it to 1 and retain 0.01 above
    # actor
    'actor_lr': 1e-3,
    'actor_beta': 0.9,
    'actor_log_std_min': -10,
    'actor_log_std_max': 2,
    'actor_update_freq': 2,
    # encoder
    'encoder_type': 'pixel',
    'encoder_feature_dim': 50,
    'encoder_lr': 1e-3,
    'encoder_tau': 0.05,
    'num_layers': 4,
    'num_filters': 32,
    'curl_latent_dim': 128,
    # sac
    'discount': 0.99,
    'init_temperature': 0.1,
    'alpha_lr': 1e-4,
    'alpha_beta': 0.5,
    'alpha_fixed': False,
    # misc
    'seed': 1,
    'save_tb': False,
    'save_buffer': False,
    'save_video': False,
    'save_model': False,
    'detach_encoder': False,
    'log_interval': 1
}


def get_TOPDM_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_tag', type=str, default="default")
    parser.add_argument('--env', type=str, default="EggHandOver-v0")
    parser.add_argument('--tau', type=int, default=30) # 20
    parser.add_argument('--num_samples_per_it', type=int, default=200) # 1000
    parser.add_argument('--frac_best', type=float, default=0.05)
    parser.add_argument('--num_iterations', type=int, default=5) # 20
    parser.add_argument('--init_noise', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.7)
    parser.add_argument('--later_noise', type=float, default=0.3)
    parser.add_argument('--later_noise_frac', type=float, default=0.3)
    parser.add_argument('--forget_init_past_ac_frac', type=float, default=0.0)
    parser.add_argument('--no_sum_rewards', dest='sum_rewards', action='store_false')
    parser.add_argument('--initialise_frac_with_prev_best', type=float, default=1.0)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--traj_len', type=int, default=50)
    parser.add_argument('--tau_scaler', type=float, default=1.0)
    parser.set_defaults(sum_rewards=True)
    args = parser.parse_args([])
    return args