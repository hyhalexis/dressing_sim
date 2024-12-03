import os
import re
from statistics import mean

# Folder containing the files
folder_path = "/scratch/alexis/data/traj_data_1124"

# Initialize dictionaries to store `ww` values for each category
p1_ww_values = []
p2_ww_values = []

# Regex pattern to extract the naming components
pattern = r"^(p1|p2).*?_([0-9.]+)\.pkl$"

# Iterate through files in the folder
for filename in os.listdir(folder_path):
    match = re.match(pattern, filename)
    if match:
        category, ww = match.groups()
        ww = float(ww)  # Convert `ww` to a float
        if category == "p1":
            p1_ww_values.append(ww)
        elif category == "p2":
            p2_ww_values.append(ww)

# Compute averages
p1_avg = mean(p1_ww_values) if p1_ww_values else None
p2_avg = mean(p2_ww_values) if p2_ww_values else None

# Print results
print(f"Average ww for p1 files: {p1_avg}")
print(f"Average ww for p2 files: {p2_avg}")
