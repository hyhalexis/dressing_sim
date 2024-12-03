import os
import re

# Directory containing the files
directory = "/scratch/alexis/data/traj_data_1124"

# Initialize dictionaries to hold the sum and count for each group
sums = {"p1_motion0": 0, "p2_motion0": 0}
counts = {"p1_motion0": 0, "p2_motion0": 0}

# Regular expression to match filenames and extract groups and the last value
pattern = re.compile(r"^(p[12]_motionq).*_([0-9.-]+)\.pkl$")

# Iterate over files in the directory
for filename in os.listdir(directory):
    match = pattern.match(filename)
    if match:
        group = match.group(1)  # p1_motion0 or p2_motion0
        last_value = float(match.group(2))  # Extract the last value
        sums[group] += last_value
        counts[group] += 1

# Compute and print the averages
for group in sums:
    if counts[group] > 0:
        average = sums[group] / counts[group]
        print(f"Average for {group}: {average:.6f}")

    else:
        print(f"No files found for {group}")
