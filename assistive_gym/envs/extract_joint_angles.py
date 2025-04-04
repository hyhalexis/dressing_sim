import re
import os

# Directory containing the files
directory = "/project_data/held/ahao/data/2025-0306-pybullet-eval-baselines/eval-all-baseline+ours-0.01_thresh_for_fcvp_mid-fric-03_07_21_25_10-000/gifs"

# Lists to store extracted values
e_values = []
s_values = []

# Regular expression pattern to match "e-95.4" and "s79.53"
pattern = re.compile(r"e([-+]?\d*\.\d+|\d+)_s([-+]?\d*\.\d+|\d+)")

# Iterate through files in the directory
for filename in os.listdir(directory):
    match = pattern.search(filename)
    if match:
        e_val = float(match.group(1))
        s_val = float(match.group(2))

        # Ensure unique (e_val, s_val) pairs
        if (e_val, s_val) not in zip(e_values, s_values):
            e_values.append(e_val)
            s_values.append(s_val)

print("e_values:", e_values)
print("s_values:", s_values)
