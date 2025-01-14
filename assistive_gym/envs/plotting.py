import os
import json
import matplotlib.pyplot as plt
import numpy as np

def process_log_file(filepath):
    """
    Reads a .log file, extracts 'mean upper arm ratio' from each entry (line), 
    and returns the first `max_entries` as a list.
    """
    mean_upper_arm_ratios = []
    
    
    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            try:
                entry = json.loads(line.strip())  # Parse each line as a JSON dictionary
                mean_upper_arm_ratio = entry.get("mean_upperarm_ratio", None)  # Extract the desired field
                if mean_upper_arm_ratio is not None:
                    mean_upper_arm_ratios.append(mean_upper_arm_ratio)
            except json.JSONDecodeError:
                print(f"Skipping malformed line in file {filepath}: {line.strip()}")
    
    return mean_upper_arm_ratios

def extract_legend_label(filepath):
    """
    Extracts the legend label from the file path.
    For example, from the folder `iql-training-p1-reward_model1+2-5050-11_29_02_20_06-000`,
    the label would be `p1-reward_model1+2-5050`.
    """
    folder_name = os.path.basename(os.path.dirname(filepath))
    parts = folder_name.split('-')
    if len(parts) >= 5:
        return '-'.join(parts[1:5])  # Extract `p1-reward_model1+2-5050`
    return folder_name  # Fallback: use the whole folder name


def plot_mean_upper_arm_ratios(files):
    """
    Plots the 'mean upper arm ratio' for the first `max_entries` entries of each .log file.
    """
    plt.figure(figsize=(12, 8))
    
    
    for i, filepath in enumerate(files):
        mean_upper_arm_ratios = process_log_file(filepath)
        steps = range(0, (len(mean_upper_arm_ratios)) * 10000, 10000)  # Step size of 10,000
        # legend = extract_legend_label(filepath)
        # avg = [0.7354270286069347] * len(steps)
        # avg = [0.8400874774915594] * len(steps)
        if i == 0:
            plt.plot(steps, mean_upper_arm_ratios, label='IQL-force+reconstruct')
        # else:
        #     plt.plot(steps, [0]+mean_upper_arm_ratios, label='IQL-no_force')
   
        if i == 0:
            avg = [0.83] * 17
            plt.plot(steps, avg, '--', label='dataset avg')
    
    plt.title(f"Mean Upper Arm Ratio")
    plt.xlabel("Step")
    plt.ylabel("Mean Upper Arm Ratio")
    plt.yticks(np.arange(0, 1.1, 0.1))


    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('fine-tune-force-reconstr.png')
    plt.show()

if __name__ == "__main__":
    # Replace with paths to your 6 .log files
    log_files = [ 
        "/scratch/alexis/data/2025-0113-pybullet-from-scratch/iql-training-from-scratch-force-reconstr-one-hot-01_13_04_36_57-000/eval.log"
                 ]
    
    # Ensure files exist before processing
    missing_files = [f for f in log_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: The following files were not found: {missing_files}")
    else:
        plot_mean_upper_arm_ratios(log_files)
