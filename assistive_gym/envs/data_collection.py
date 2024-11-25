import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

conda_env_name = "assistive_gym"
path_to_conda = '/home/alexishao/anaconda3/condabin/conda'
conda_sh_path = '/home/alexishao/anaconda3/etc/profile.d/conda.sh'

original_commands = [
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 2 --motion_id 0 --camera_pos front --garment_id 1", 
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 2 --motion_id 2 --camera_pos front --garment_id 1",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 2 --motion_id 4 --camera_pos front --garment_id 1",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 2 --motion_id 5 --camera_pos front --garment_id 1",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 2 --motion_id 6 --camera_pos front --garment_id 1",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 2 --motion_id 7 --camera_pos front --garment_id 1",

    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 1 --motion_id 0 --camera_pos front --garment_id 1",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 1 --motion_id 2 --camera_pos front --garment_id 1",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 1 --motion_id 4 --camera_pos front --garment_id 1",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 1 --motion_id 5 --camera_pos front --garment_id 1",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 1 --motion_id 6 --camera_pos front --garment_id 1",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 1 --motion_id 7 --camera_pos front --garment_id 1",

    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 2 --motion_id 0 --camera_pos front --garment_id 2",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 2 --motion_id 2 --camera_pos front --garment_id 2",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 2 --motion_id 4 --camera_pos front --garment_id 2",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 2 --motion_id 5 --camera_pos front --garment_id 2",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 2 --motion_id 6 --camera_pos front --garment_id 2",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 2 --motion_id 7 --camera_pos front --garment_id 2",

    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 1 --motion_id 0 --camera_pos front --garment_id 2",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 1 --motion_id 2 --camera_pos front --garment_id 2",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 1 --motion_id 4 --camera_pos front --garment_id 2",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 1 --motion_id 5 --camera_pos front --garment_id 2",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 1 --motion_id 6 --camera_pos front --garment_id 2",
    f"{path_to_conda} run -n {conda_env_name} python launch_eval.py --policy 1 --motion_id 7 --camera_pos front --garment_id 2",
]

repeated_commands = [cmd for cmd in original_commands for _ in range(20)]

def run_command_on_gpu(command, gpu_id):
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    result = subprocess.run(command, shell=True, env=env, capture_output=True, text=True)
    return (command, result.stdout, result.stderr, result.returncode)

def main():
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(run_command_on_gpu, cmd, i % 2) for i, cmd in enumerate(repeated_commands)]
        
        for future in as_completed(futures):
            command, stdout, stderr, returncode = future.result()
            if returncode == 0:
                print(f"Command '{command}' completed successfully.")
                print(stdout)
            else:
                print(f"Command '{command}' failed with error:")
                print(stderr)

if __name__ == "__main__":
    main()
