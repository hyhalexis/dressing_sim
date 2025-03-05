import os
import re
from collections import defaultdict

# Folder containing the files
folder_path = "/project_data/held/ahao/data/2025-0228-pybullet-eval-baselines/eval-all-baseline+ours-03_03_07_01_38-000/gifs"

# Data structure to store the sums and counts
data = defaultdict(lambda: {'sum': 0, 'count': 0})

# Regular expression to parse the filenames
pattern = re.compile(r"(p\d+)_motion(\d+).*_(\d+\.\d+)\.gif")

# Parse filenames in the folder
for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        policy = match.group(1)
        motion = match.group(2)
        last_number = float(match.group(3))
        
        # Update sum and count for this policy and motion
        key = (policy, motion)
        data[key]['sum'] += last_number
        data[key]['count'] += 1


# Compute averages
averages = {key: values['sum'] / values['count'] for key, values in data.items()}

# Display results
for (policy, motion), avg in averages.items():
    print(f"{policy} motion{motion}: {avg:.6f}")

import os
import re
from collections import defaultdict

# Folder containing the files
folder_path =  "/project_data/held/ahao/data/2025-0228-pybullet-eval-baselines/eval-all-baseline+ours-03_03_07_01_38-000/gifs"

# Data structure to store the sums and counts for each participant
participant_data = defaultdict(lambda: {'sum': 0, 'count': 0})

# Regular expression to parse the filenames
pattern = re.compile(r"(p\d+)_motion\d+.*_(\d+\.\d+)\.gif")

# Parse filenames in the folder
for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        participant = match.group(1)
        last_number = float(match.group(2))
        
        # Update sum and count for this participant
        participant_data[participant]['sum'] += last_number
        participant_data[participant]['count'] += 1

# Compute averages
participant_averages = {
    participant: values['sum'] / values['count'] for participant, values in participant_data.items()
}

# Sort results by participant and display
for participant, avg in sorted(participant_averages.items()):
    print(f"{participant}: {avg:.6f}")
