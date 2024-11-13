import time
import torch
import click
import socket
from chester.run_exp import run_experiment_lite, VariantGenerator
from softgym.registered_env import env_arg_dict
from dressing_motion.curl.train import run_task

def get_sa_mlp_list(layer):
    if layer == 3:
        return  [[64, 64, 128], [128, 128, 256], [256, 512, 1024]]
    elif layer == 4:
        return [[64, 64, 128], [128, 128, 256], [256, 256, 512], [512, 512, 1024]]
    elif layer == 5:
        return [[64, 64, 128], [128, 128, 256], [256, 256, 512], [512, 512, 1024], [1024, 1024, 2048]]

def get_fp_mlp_list(layer):
    if layer == 3:
        return  [[256, 256], [256, 128], [128, 128, 128]]
    elif layer == 4:
        return  [[256, 256], [256, 256], [256, 256, 128], [128, 128, 128]]
    elif layer == 5:
        return [[256, 256], [256, 256], [256, 256, 128], [128, 128, 128], [128, 128, 128]]
    
def get_fp_k(layer):
    if layer == 3:
        return  [1, 3, 3]
    elif layer == 4:
        return  [3, 3, 3, 3]
    elif layer == 5:
        return  [3, 3, 3, 3, 3]

def get_linear_mlp_list(layer, encoder_type):
    if 'flow' in encoder_type:
        if layer == 3:
            return  [128, 128]
        elif layer == 4:
            return  [128, 128, 128]
        elif layer == 5:
            return  [128, 128, 128]
    else:
        if layer == 3:
            return  [256, 256, 256, 128, 128, 128, 128, 128, 128]

def get_sa_ratio(layer, observation_mode):
    if observation_mode == 'pointcloud':
        if layer == 3:
            return  [0.4, 0.5]
        elif layer == 4:
            return [0.4, 0.5, 0.6]
        elif layer == 5:
            return [0.4, 0.5, 0.6, 0.7]
    else:
        if layer == 3:
            return [1, 1]

def get_sa_radius(layer):
    if layer == 3:
        return [0.05, 0.1]
        # return [0.2, 0.4]
    elif layer == 4:
        return [0.05, 0.1, 0.2]
    elif layer == 5:
        return [0.05, 0.1, 0.2, 0.4]

def get_batch_size(obs_mode, encoder_type, voxel_size, env_name, load_real_world_buffer_path):
    return 64

def get_k(observation_mode):
    if observation_mode == 'reduced_pointcloud':
        return 6
    elif observation_mode == 'pointcloud':
        # return 30
        return 6
    elif observation_mode == 'pointcloud_2':
        return 6

def get_force_w(load_real_world_buffer_path):
    if load_real_world_buffer_path is not None:
        return 0
    else:
        return 0.001


def get_train_configs(replay_buffer_num, cached_state_path):
    if replay_buffer_num == 1:
        # for training universal policy on all pose regions
        if cached_state_path == 'dressing_rotation_aligned_partition_all.pkl':
            res = []
            for i in range(27):
                res += [j for j in range(i*50, i*50 + 40)]
            # print(res)
            return res
        else:
            print("train configs are 0-40")
            return [i for i in range(40)]
    else:
        # this is distillation 
        if cached_state_path == 'dressing_rotation_aligned_partition_all.pkl':
            return [i for i in range(0, 40)]


def get_center_r_w(real_world_buffer):
    if real_world_buffer is not None:
        return 0
    else:
        return 0.02


def get_center_p_w(real_world_buffer):
    if real_world_buffer is not None:
        return 0
    else:
        return 0.05

def get_collision_w(real_world_buffer):
    if real_world_buffer is not None:
        return 0
    else:
        return 0.01

def get_pc_feature_dim(obs):
    if obs == 'real_partial_pc':
        return 2
    if obs == 'pointcloud_3':
        return 3


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    # NOTE some example experiments to run:
    # 1. Train a policy using just RL, with occluded observation, on region 13.
    # should set use_distillation to False, and train_regions to [13], and eval_regions to [13]. observation_mode to 'real_partial_pc'
    # set all the randomization parameters to False.
    # 2. Train a policy using distillation, with occluded observation, on region 13.
    # should set use_distillation to True, and train_regions to [13], and eval_regions to [13]. observation_mode to 'real_partial_pc'
    # set all the randomization parameters to False. set use_teacher_bc_loss to True.
    # should also have the teacher_actors_path to be the path to the teacher policy on region 13.
    # 3. Train a policy using distillation, with occluded observation, on all regions, where the studnet receives the randomized observations.
    # should set use_distillation to True, and train_regions to [i for i in range(27)], and eval_regions to [i for i in range(27)]. observation_mode to 'real_partial_pc'
    # set all the randomization parameters to True. set use_teacher_bc_loss to True.
    # should also have the teacher_actors_path to be the path to the teacher policy on all regions.
    
    exp_prefix = '2024-0505-test-cleaned-code-real-partial-pc-distill-27-regions'

    vg = VariantGenerator()

    vg.add('agent', ['curl_sac'])

    # training configs
    vg.add('evaluate_only', [False]) 
    vg.add('num_train_steps', [5000000])
    vg.add('seed', [100])
    vg.add('train_poses', lambda replay_buffer_num, cached_states_path: [
        # get_train_configs(replay_buffer_num, cached_states_path)
        [i for i in range(0, 45)]
    ]) 
    vg.add('eval_poses', [[i for i in range(45, 46)]]) 
    vg.add('eval_freq', [2])


    ### dressing env args
    vg.add('env_name', ["dressing"]) 
    vg.add('dual_arm', [False]) 
    vg.add('action_mode', ['rotation'])
    # NOTE: this controls the observation of the student policy. 
    # "pointcloud_3" means the unoccluded observation of the cloth and human, where the policy observes the whole arm and cloth point cloud,
    # including the arm points that are occluded by the cloth. i.e., we ignore the cloth occlusion on the human arm.
    # "real_partial_pc" means the occluded observation of the cloth and human, where the policy observes the cloth,
    # and the human arm points that are not occluded by the cloth. i.e., the cloth will have occlusion on the human arm.
    # for dealing with human motions we need to use "real_partial_pc".
    # vg.add('observation_mode', ['pointcloud_3'])
    vg.add('observation_mode', ['real_partial_pc'])
    
    vg.add('cap_reward', [True])
    vg.add('line1_extend_factor', [0])
    vg.add('force_w', lambda load_real_world_buffer_path: [get_force_w(load_real_world_buffer_path)])
    vg.add('force_threshold', [1000])
    vg.add('voxel_size', [0.00625 * 10])
    vg.add('horizon', [150]) 
    vg.add('collision_threshold', [0.01])
    vg.add('collision_w', lambda load_real_world_buffer_path: [get_collision_w(load_real_world_buffer_path)]) # 1 previously
    vg.add('no_move_collision', [True]) # False previously
    vg.add('no_move_collision_threshold', [0.012]) # 0.01 previously
    vg.add('upper_w', [5])
    vg.add('cached_states_path', ['dressing_rotation_aligned_all_poses_garments.pkl']) 
    vg.add('garments', [['hospital_gown', 'tshirt_26', 'tshirt_68', 'tshirt_4', 'tshirt_392']]) 
    vg.add('cloth_scale', [None]) 
    vg.add('center_align_reward_w', lambda load_real_world_buffer_path: [get_center_r_w(load_real_world_buffer_path)])
    vg.add('center_align_penalty_w', lambda load_real_world_buffer_path: [get_center_p_w(load_real_world_buffer_path)])
    vg.add('near_center_range', [0.03])
    vg.add('far_center_range', [0.075])
    vg.add("max_translation", [0.1]) 
    vg.add('clip_rotation_mode', ['to_yz']) 

    # for adpating to real world
    vg.add("apply_pitch_limit", [0]) 
    vg.add("apply_yaw_limit", [0])
    vg.add('frame_skip', [0])
    vg.add('load_real_world_buffer_path', [None]) # 'data/dressing_proj/real_world/buffer'

    # for robustify and not cheat and make the rotation kinematically feasible
    # NOTE: when training a randomized observation policy, please set 
    # randomize_cloth_observation, randomize_cloth_erosion, randomize_cloth_dilation, and randomize_gripper_pos to be True. Others remain False
    vg.add('randomize_camera', [False])
    vg.add('randomize_cloth_observation', [False])
    vg.add('randomize_gripper_pos', [False])
    vg.add('randomize_cloth_erosion', [False])
    vg.add('randomize_cloth_dilation', [False])
    vg.add('use_human_finger_as_cropping_offset', [True])
    vg.add('randomize_sleeve_width', [False])
    vg.add('randomize_sleeve_length', [False])
    vg.add('initial_state_randomization', [False])

    ## reward predictor parameters. 
    vg.add("train_reward_predictor", [0]) 
    vg.add("train_actor", [1]) 
    vg.add("train_critic", [1]) 
    vg.add("reward_predictor_lr", [1e-4]) 


    # distillation parameters
    vg.add("replay_buffer_num", [27]) 
    vg.add('full_obs_guide', [True]) # NOTE: should always be True
    vg.add("use_distillation", [False]) # NOTE: if set to be true, we are doing teacher-student distillation. Otherwise it's just regular RL training.
    vg.add('train_regions', [[i for i in range(27)]]) # This controls the region we want to train on. We have regions between 0-26.
    # vg.add('eval_regions', [[0, 5]])
    vg.add('eval_regions', [[i for i in range(27)]])
    vg.add("use_teacher_bc_loss", [True]) # NOTE: if using distillation, this should be True.
    vg.add("bc_loss_weight", [0.01]) 
    vg.add("distill_translation_only", [False]) 
    vg.add("RL_loss_weight", [1]) 
    vg.add("sample_replay_buffer_num", [1]) 
    vg.add("teacher_pc_feature_dim", [3]) 
    vg.add("teacher_critics_path", lambda cached_states_path: [
        # get_teacher_critic_paths(cached_states_path)
        None
    ]) 

    vg.add("teacher_actors_path", lambda cached_states_path: [ # NOTE: these are the teacher policies that we load to train the student policy. 
       []
        # ["data/dressing_proj/5_garment_regional_teachers/0/actor_best_test_1210132_0.84764.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/1/actor_best_test_1240039_0.8294.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/2/actor_1320047.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/3/actor_740002.pt",
        #     "data/dressSing_proj/5_garment_regional_teachers/4/actor_best_test_920000_0.84499.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/5/actor_best_test_600079_0.79381.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/6/actor_1520026.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/7/actor_best_test_1040001_0.88014.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/8/actor_best_test_670003_0.70485.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/9/actor_best_test_890037_0.71538.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/10/actor_best_test_890023_0.8178.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/11/actor_best_test_830149_0.83556.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/12/actor_best_test_660125_0.65101.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/13/actor_best_test_1960083_0.740.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/14/actor_1070002.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/15/actor_best_test_610049_0.76728.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/16/actor_best_test_560048_0.77099.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/17/actor_best_test_630041_0.79704.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/18/actor_best_test_560101_0.73981.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/19/actor_best_test_570131_0.8338.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/20/actor_best_test_940055_0.73806.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/21/actor_best_test_640015_0.72003.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/22/actor_best_test_520140_0.78158.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/23/actor_best_test_550014_0.7906.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/24/actor_best_test_920000_0.78185.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/25/actor_best_test_690024_0.64553.pt",
        #     "data/dressing_proj/5_garment_regional_teachers/26/actor_best_test_750006_0.701.pt"]
    ]) 

    # resume experiments or loading pretrained models
    vg.add('resume_from_ckpt', [False])
    vg.add('resume_from_exp_path', [None])
    vg.add('resume_from_path_actor', [None])
    vg.add('resume_from_path_critic', [None])
    vg.add('actor_load_name', [
        '/home/alexishao/dressing_ws/softagent_rpad/ckpt/actor_best_test_600023_0.65914.pt'
    ])
    vg.add('critic_load_name', [
        None
    ])

    # save stuff
    vg.add('save_explore_imgs', [False]) 
    vg.add('save_buffer', [False]) 
    vg.add('saving_gif_start', [200000])
    vg.add('use_wandb', [0])
    vg.add('save_tb', [False])
    vg.add('save_video', [True])
    vg.add('save_model', [True])

    # policy architecture parameters
    vg.add('cuda_idx', [0])
    vg.add('encoder_type', ['pointcloud_flow'])
    vg.add("vector_q_mode", ['repeat_action']) 
    vg.add("flow_q_mode", ['repeat_gripper']) 

    # RL algorithm
    vg.add('detach_encoder', [False])
    vg.add('replay_buffer_capacity', [400000])
    vg.add('init_steps', [0])
    vg.add('critic_lr', [1e-4]) # 1e-4
    vg.add('encoder_lr', [1e-6])
    vg.add('actor_lr', [1e-4]) # 1e-4
    vg.add('alpha_lr', [1e-4]) # 1e-4
    vg.add('actor_update_freq', [4])
    vg.add('batch_size', lambda observation_mode, encoder_type, voxel_size, env_name, load_real_world_buffer_path: [
        get_batch_size(observation_mode, encoder_type, voxel_size, env_name, load_real_world_buffer_path)
    ])
    vg.add('lr_decay', [None])
    

    # PointNet++ parameters
    vg.add('pc_feature_dim', lambda observation_mode: [get_pc_feature_dim(observation_mode)])
    vg.add('pc_num_layers', [3])
    vg.add('sa_radius', lambda pc_num_layers: [get_sa_radius(pc_num_layers)])
    vg.add('sa_ratio', lambda pc_num_layers, observation_mode: [get_sa_ratio(pc_num_layers, observation_mode)])
    vg.add('sa_mlp_list', lambda pc_num_layers: [get_sa_mlp_list(pc_num_layers)])
    vg.add('linear_mlp_list', lambda pc_num_layers, encoder_type: [get_linear_mlp_list(pc_num_layers, encoder_type)])
    vg.add('fp_mlp_list', lambda pc_num_layers: [get_fp_mlp_list(pc_num_layers)])
    vg.add('fp_k', lambda pc_num_layers: [get_fp_k(pc_num_layers)])

    if not debug:
        pass
    else:
        exp_prefix += '_debug'

    print('Number of configurations: ', len(vg.variants()))
    print("exp_prefix: ", exp_prefix)

    hostname = socket.gethostname()
    gpu_num = torch.cuda.device_count()

    variations = set(vg.variations())
    task_per_gpu = 1
    all_vvs = vg.variants()
    slurm_nums = len(all_vvs) // task_per_gpu
    if len(all_vvs) % task_per_gpu != 0:
        slurm_nums += 1

    sub_process_popens = []
    for idx in range(slurm_nums):
        beg = idx * task_per_gpu
        end = min((idx+1) * task_per_gpu, len(all_vvs))
        vvs = all_vvs[beg:end]
    # for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 10:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        if mode in ['seuss', 'autobot']:
            if idx == 0:
                # compile_script = 'compile.sh'  # For the first experiment, compile the current softgym
                compile_script = None  # For the first experiment, compile the current softgym
                wait_compile = 0
            else:
                compile_script = None
                wait_compile = 0  # Wait 30 seconds for the compilation to finish
        elif mode == 'ec2':
            compile_script = 'compile_1.0.sh'
            wait_compile = None
        else:
            compile_script = wait_compile = None
        if hostname.startswith('autobot') and gpu_num > 0:
            env_var = {'CUDA_VISIBLE_DEVICES': str(idx % gpu_num)}
        else:
            env_var = None
        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variants=vvs,
            # variant=vv,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
            env=env_var,
            variations=variations,
            task_per_gpu=task_per_gpu
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
