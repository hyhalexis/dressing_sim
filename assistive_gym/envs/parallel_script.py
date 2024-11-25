import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import pybullet as p
from .utils import *
import argparse
from chester.run_exp import run_experiment_lite, VariantGenerator


# env = None
def get_cost(args):
    cur_state, action_trajs, env_class, env_kwargs, worker_i = args
    # global env
    # if env is None:
    #     # Need to create the env inside the function such that the GPU buffer is associated with the child process and avoid any deadlock.
    #     # Use the global variable to access the child process-specific memory

    # import pdb; pdb.set_trace()
    print("env ", env_kwargs)
    env = env_class(**env_kwargs)
    env.reset()

    N = action_trajs.shape[0]
    costs = []
    for i in range(N):
        # print(worker_i, f'{i}/{N}')
        load_env(env, state=cur_state)
        ret = 0
        rewards = []
        for action in action_trajs[i, :]:
            _, reward, _, _ = env.step(action)
            ret += reward
            rewards.append(reward)
        costs.append([-ret, rewards])
        # print('get_cost {}: {}'.format(i, ret))
    p.disconnect(env.id)
    return costs


class ParallelRolloutWorker(object):
    """ Rollout a set of trajectory in parallel. """

    def __init__(self, env_class, env_kwargs, plan_horizon, action_dim, num_worker=32):
        self.num_worker = num_worker
        self.plan_horizon, self.action_dim = plan_horizon, action_dim
        self.env_class, self.env_kwargs = env_class, env_kwargs
        self.pool = Pool(processes=num_worker)

    def cost_function(self, init_state, action_trajs):
        action_trajs = action_trajs.reshape([-1, self.plan_horizon, self.action_dim])
        splitted_action_trajs = np.array_split(action_trajs, self.num_worker)
        ret = self.pool.map(get_cost, [(init_state, splitted_action_trajs[i], self.env_class, self.env_kwargs, i) for i in range(self.num_worker)])
        # ret = get_cost((init_state, action_trajs, self.env_class, self.env_kwargs))
        flat_costs = [item for sublist in ret for item in sublist]  # ret is indexed first by worker_num then traj_num
        return flat_costs
    
def main(args):
    exp_prefix = '1118_data_collection'

    vg = VariantGenerator()

    vg.add('variant_path', ['/home/alexishao/assistive-gym-dressing/assistive-gym-fem/assistive_gym/envs/variant.json'])

    if args.policy == '1':
        vg.add('actor_load_name', ['/home/alexishao/assistive-gym-dressing/assistive-gym-fem/assistive_gym/envs/ckpt/actor_1900106.pt'])
        vg.add('horizon', [250])

    else:
        vg.add('actor_load_name', ['/home/alexishao/assistive-gym-dressing/assistive-gym-fem/assistive_gym/envs/ckpt/actor_best_test_600023_0.65914.pt'])
        vg.add('horizon', [170])


    vg.add('observation_mode', ['real_partial_pc'])
    # vg.add('observation_mode', ['pointcloud_3'])
    # vg.add('real_partial_pc_voxel_size', [0.00625 * 4])
    vg.add('max_translation', [0.1])
    vg.add('frame_skip', [0])
    vg.add('eval_poses', [[i for i in range(30)]])
    vg.add('plot_gif', [True])
    vg.add('plot_gradients', [False])
    vg.add('save_images', [False])
    vg.add('draw_pc', [False])
    vg.add('decompose_picker_action', [False])
    vg.add('train_reward_predictor', [False])
    if args.render:
        vg.add('render', [True])

    else:
        vg.add('render', [False])
    vg.add('action_scale', [0.025]) # 0.025
    vg.add('camera_pos', [args.camera_pos])
    vg.add('motion_id', [args.motion_id])
    vg.add('garment_id', [args.garment_id])
    vg.add('policy', [args.policy])
    vg.add('rand', [args.rand])

    if not args.debug:
        pass
    else:
        exp_prefix += '_debug'

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
        if args.mode in ['seuss', 'autobot']:
            if idx == 0:
                # compile_script = 'compile.sh'  # For the first experiment, compile the current softgym
                compile_script = None  # For the first experiment, compile the current softgym
                wait_compile = 0
            else:
                compile_script = None
                wait_compile = 0  # Wait 30 seconds for the compilation to finish
        elif args.mode == 'ec2':
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
            mode=args.mode,
            dry=args.dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=args.debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
            env=env_var,
            variations=variations,
            task_per_gpu=task_per_gpu
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if args.debug:
            break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--mode', type=str, default='local', 
        choices=['local', 'seuss', 'autobot', 'ec2'], 
    )
    
    parser.add_argument(
        '--debug', action='store_true', 
    )
    
    parser.add_argument(
        '--dry', action='store_true', 
    )

    parser.add_argument('--policy', default=2, choices=['1', '2'])
    parser.add_argument('--camera_pos', default='side', choices=['front', 'side'])

    parser.add_argument('--motion_id', default=1)
    parser.add_argument('--garment_id', default=1)
    parser.add_argument('--render', default=0)
    parser.add_argument('--rand', default=0)
    
    # Parse arguments
    args = parser.parse_args()



    # Can be used to benchmark the system
    from manipulation.sim import SimpleEnv
    import copy
    from RL.train_RL_api import default_config
    import pickle

    task_config = "gpt_4/data/parsed_configs_semantic_articulated/test_without_table.yaml"

    config = copy.deepcopy(default_config)
    config['config_path'] = task_config
    config['gui'] = False
    env = SimpleEnv(**config)
    env.reset()

    env_class = SimpleEnv
    env_kwargs = config
    initial_state = "manipulation/gpt_tasks/Load_Dishes_into_Dishwasher/12594/RL/open_the_dishwasher_door/best_final_state.pkl"
    with open(initial_state, 'rb') as f:
        initial_state = pickle.load(f)

    action_trajs = []
    for i in range(700):
        action = env.action_space.sample()
        action_trajs.append(action)
    action_trajs = np.array(action_trajs)
    rollout_worker = ParallelRolloutWorker(env_class, env_kwargs, 10, 7)
    cost = rollout_worker.cost_function(initial_state, action_trajs)
    print('cost:', cost)