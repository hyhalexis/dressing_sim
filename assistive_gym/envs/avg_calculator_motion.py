import re
from collections import defaultdict
import os

# List of filenames
folder_name = "/scratch/alexis/data/traj_data_with_force/gifs"

# Initialize a dictionary to store motion numbers and their values
motion_data = defaultdict(list)

# Pattern to match p2 files and extract motion and last number
pattern = re.compile(r"p2_motion(\d+).*?_([-+]?\d+\.\d+)\.gif")

# Process filenames
for filename in os.listdir(folder_name):
    # if "0.0_0.0.pkl" in filename:
    #     continue
    match = pattern.search(filename)
    if match:
        motion = int(match.group(1))  # Extract motion number
        last_number = float(match.group(2))  # Extract last number
        motion_data[motion].append(last_number)

# Compute averages
averages = {motion: sum(values) / len(values) for motion, values in motion_data.items()}

# Print results
for motion, avg in sorted(averages.items()):
    print(f"Motion {motion}: Average of last number = {avg}")
