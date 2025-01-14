import os
import re
from collections import defaultdict

# Function to extract pose and the last number from a filename
def extract_pose_and_value(filename):
    match = re.match(r"pose(\d+)_.*_(\d+\.\d+)\.gif", filename)
    # match = re.match(r".*step(\d+)_.*?_(0\.\d+)\.gif", filename)
    if match:
        pose = int(match.group(1))  # Pose number (e.g. 0, 1, ..., 27)
        value = float(match.group(2))  # Last number in the filename
        return pose, value
    return None, None

# Function to calculate averages for each step
def calculate_averages(folder_path):
    step_values = defaultdict(lambda: defaultdict(list))  # {step: {pose_group: values}}

    # Define poses to skip
    skip_poses = {17, 26}

    # Iterate through all files in the folder
    for filename in sorted(os.listdir(folder_path)):
        pose, value = extract_pose_and_value(filename)
        if pose is not None and pose not in skip_poses:  # Skip specified poses
            # Extract the step number from the filename
            match = re.search(r"step(\d+)", filename)
            if match:
                step = int(match.group(1))  # Step number (0-4)
                # Add the value to the corresponding step and pose group
                step_values[step][pose // 6].append(value)  # Group poses in sets of 6

    # Calculate and print averages
    averages = {}
    for step, pose_groups in step_values.items():
        step_averages = []
        for pose_group, values in pose_groups.items():
            avg_value = sum(values) / len(values) if values else 0
            step_averages.append((pose_group, avg_value))
        averages[step] = step_averages

    # Print results
    for step, step_averages in averages.items():
        print(f"Step {step}:")
        for pose_group, avg_value in step_averages:
            print(f"  Pose group {pose_group * 6}-{(pose_group + 1) * 6 - 1}: Average = {avg_value:.4f}")

# Specify the folder containing the files
folder_path = '/scratch/alexis/data/2024-1210-pybullet-eval-ckpt/eval-static-poses-12_11_19_51_05-000/gifs'

# Call the function to calculate averages
calculate_averages(folder_path)
