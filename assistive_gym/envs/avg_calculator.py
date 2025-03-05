import os
import re
from statistics import mean

# Folder containing the files
# folder_path = "/scratch/alexis/data/2024-1222-pybullet-eval-ckpt/eval-p1-force-with-joint-rand-12_23_00_29_54-000/gifs"
folder_path = "/project_data/held/ahao/data/traj_data_with_one-hot_reconstr_flex-72585/gifs"
# Initialize dictionaries to store `ww` values for each category
p1_ww_values = []
p2_ww_values = []
all_values = []

# Regex pattern to extract the naming components
pattern = r"^(p0|p1|p2|p3).*?_([0-9.]+)\.gif$"
# pattern = re.compile(r"^(p[12])_.*tshirt_26.*_(\d*\.\d+)\.pkl$")


# Iterate through files in the folder
for filename in os.listdir(folder_path):
    # if filename.endswith('0.0_0.0.pkl'):
    #     continue
    match = re.match(pattern, filename)
    if match:
        category, ww = match.groups()
        ww = float(ww)  # Convert `ww` to a float
        if category == "p1":
            p1_ww_values.append(ww)
        elif category == "p2":
            p2_ww_values.append(ww)
        all_values.append(ww)

# Compute averages
p1_avg = mean(p1_ww_values) if p1_ww_values else None
p2_avg = mean(p2_ww_values) if p2_ww_values else None

# Print results
print(f"Average ww for p1 files: {p1_avg}")
print(f"Average ww for p2 files: {p2_avg}")
print(mean(all_values))
