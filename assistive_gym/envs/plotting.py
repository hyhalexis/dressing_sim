import os
import json
import matplotlib.pyplot as plt

def process_log_file(filepath, max_entries=10, init_value=0):
    """
    Reads a .log file, extracts 'mean upper arm ratio' from each entry (line), 
    and returns the first `max_entries` as a list.
    """
    mean_upper_arm_ratios = [init_value]
    
    
    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            if i >= max_entries:
                break  # Limit to `max_entries` entries
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


def plot_mean_upper_arm_ratios(files, max_entries=10):
    """
    Plots the 'mean upper arm ratio' for the first `max_entries` entries of each .log file.
    """
    plt.figure(figsize=(12, 8))
    
    for i, filepath in enumerate(files):
        init_value = 0.543980349063669 if "iql-training-p1" in filepath else 0.494416040246595 if "iql-training-p2" in filepath else None
        mean_upper_arm_ratios = process_log_file(filepath, max_entries=max_entries, init_value=init_value)
        steps = range(0, len(mean_upper_arm_ratios) * 10000, 10000)  # Step size of 10,000
        legend = extract_legend_label(filepath)
        plt.plot(steps, mean_upper_arm_ratios, label=legend)
    
    plt.title(f"Mean Upper Arm Ratio (First {max_entries} Entries)")
    plt.xlabel("Step")
    plt.ylabel("Mean Upper Arm Ratio")

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace with paths to your 6 .log files
    log_files = [
        "/scratch/alexis/data/2024-1128-pybullet-finetuning-simple/iql-training-p1-reward_model1+2-7030-11_29_02_15_47-000/eval.log",
        "/scratch/alexis/data/2024-1128-pybullet-finetuning-simple/iql-training-p1-reward_model1+2-5050-11_29_02_20_06-000/eval.log",
        "/scratch/alexis/data/2024-1128-pybullet-finetuning-simple/iql-training-p1-reward_model1-only-11_29_02_23_08-000/eval.log",
        "/scratch/alexis/data/2024-1126-pybullet-finetuning-simple/iql-training-p2-reward_model1+2-7030-11_26_19_44_29-000/eval.log",
        "/scratch/alexis/data/2024-1126-pybullet-finetuning-simple/iql-training-p2-reward_model1+2-5050-11_27_00_48_02-000/eval.log",
    ]
    
    # Ensure files exist before processing
    missing_files = [f for f in log_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: The following files were not found: {missing_files}")
    else:
        plot_mean_upper_arm_ratios(log_files, max_entries=10)
