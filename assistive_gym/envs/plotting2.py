import os
import re
import numpy as np
import matplotlib.pyplot as plt

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

# List of folder paths
folder_paths = [
    # "/scratch/alexis/data/2024-1205-pybullet-finetuning/iql-training-p1-reward_model1-only-12_05_07_16_53-000/gifs",
    # "/scratch/alexis/data/2024-1205-pybullet-finetuning/iql-training-p2-reward_model1-only-12_05_05_50_38-000/gifs",
    # "/scratch/alexis/data/2024-1206-pybullet-from-scratch/iql-training-p0-reward_model1-only-12_06_15_07_34-000/gifs",
    # # "/scratch/alexis/data/2024-1205-pybullet-finetuning/iql-training-p1-reward_model1-only-12_05_07_16_53-000/gifs_slow",
    # # "/scratch/alexis/data/2024-1205-pybullet-finetuning/iql-training-p2-reward_model1-only-12_05_05_50_38-000/gifs_slow"
    # "/scratch/alexis/data/2024-1209-pybullet-eval-ckpt/gifs"
    "/scratch/alexis/data/traj_data_with_force/gifs"


]
# Regular expression to extract step and last number
pattern = re.compile(r".*step(\d+)_.*?_(0\.\d+)\.gif")
# pattern = re.compile(r".*tshirt_26.*step(\d+)_.*?_(0\.\d+)\.gif")
# pattern = re.compile(r"^(?!.*motion7).*tshirt_68.*step(\d+)_.*?_(0\.\d+)\.gif")


# Dictionary to hold steps and averages for each folder
folder_results = {}

# Process each folder
for i in range(len(folder_paths)):
    folder_path = folder_paths[i]
    step_groups = {}
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist. Skipping...")
        continue

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # if "motion7" in filename:
        #     continue
        match = pattern.search(filename)
        if match:
            step = int(match.group(1))
            value = float(match.group(2))

            if step not in step_groups:
                step_groups[step] = []
            step_groups[step].append(value)
        
    # Compute the averages for each step group
    if step_groups:
        steps = sorted(step_groups.keys())
        averages = [np.mean(step_groups[step]) for step in steps]
        folder_results[folder_path] = (steps, averages)
    else:
        print(f"No matching files found in folder {folder_path}. Skipping...")

# Plotting results for all folders
plt.figure(figsize=(10, 6))
# legends = ['p1_reward-model1', 'p2_reward-model1', 'from-scratch_reward-model1', 'p1_reward-model1_slow-motion', 'p2_reward-model1_slow-motion', ]
legends = ['poses-from-motions']


i = 0
for folder_path, (steps, averages) in folder_results.items():
    # legend = extract_legend_label(folder_path)
    legend = legends[i]
    plt.plot(steps, averages, marker='o', linestyle='-', label=legend)
    i+=1

# Add plot details
plt.xlabel("Step")
plt.ylabel("Average Value")
plt.title("Upper Arm Dressed Ratios")
plt.grid(True)
plt.legend(loc="best")

# Optional: Save the plot
plt.savefig("average_values_by_step_multiple_folders.png", dpi=300)

plt.show()