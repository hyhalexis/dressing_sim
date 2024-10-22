import time
import torch
import click
import socket
from chester.run_exp import run_experiment_lite, VariantGenerator
from eval import run_task
# from env_viewer import viewer


def get_all_eval_configs():
    res = []
    for i in range(27):
        res = res + [j for j in range(i*50 + 40, i*50 + 50)]
    # print(res)
    return res

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    # viewer('DressingSawyerHuman-v1')
    exp_prefix = '0930_first_exp'

    vg = VariantGenerator()

    vg.add('variant_path', ['/home/alexishao/assistive-gym-dressing/assistive-gym-fem/assistive_gym/envs/variant.json'])
    vg.add('actor_load_name', ['/home/alexishao/assistive-gym-dressing/assistive-gym-fem/assistive_gym/envs/ckpt/actor_1900106.pt'])
    
    vg.add('observation_mode', ['real_partial_pc'])
    # vg.add('observation_mode', ['pointcloud_3'])
    # vg.add('real_partial_pc_voxel_size', [0.00625 * 4])
    vg.add('max_translation', [0.1])
    vg.add('frame_skip', [0])
    vg.add('eval_poses', [[i for i in range(30)]])
    vg.add('plot_gif', [True])
    vg.add('horizon', [150])
    vg.add('plot_gradients', [False])
    vg.add('save_images', [False])
    vg.add('draw_pc', [False])
    vg.add('decompose_picker_action', [False])
    vg.add('train_reward_predictor', [False])

    # vg.add('randomize_cloth_observation', [False])
    # vg.add('gripper_crop_lower_y', [0])
    # vg.add('gripper_crop_higher_y', [0])
    # vg.add('gripper_crop_higher_x', [0])
    # vg.add('include_tool_pc', [1])
    # vg.add('randomize_gripper_pos', [0])
    # vg.add('cloth_depth_erosion', [0])
    # vg.add('cloth_depth_dilation', [0])

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
