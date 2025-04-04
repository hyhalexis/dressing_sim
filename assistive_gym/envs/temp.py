import os
import re

# Set the folder path
folder_path = "/project_data/held/ahao/data/2025-0306-pybullet-eval-baselines/eval-all-baseline+ours-0.01_thresh_for_fcvp_new_cloth_params-03_06_05_16_43-000/gifs"

# Regular expression to match "e-xx_szz"
pattern = re.compile(r"e-([\d\.]+)_s([\d\.]+)")

# Lists to store corresponding xx and zz values
xx_list = []
zz_list = []

# Set to track unique pairs
unique_pairs = set()

# Iterate through files in the folder
for filename in os.listdir(folder_path):
    match = pattern.search(filename)
    if match:
        xx = float(match.group(1))
        zz = float(match.group(2))
        pair = (xx, zz)
        
        if pair not in unique_pairs:  # Ensure uniqueness
            unique_pairs.add(pair)
            xx_list.append(xx)
            zz_list.append(zz)

print("xx values:", xx_list)
print("zz values:", zz_list)
