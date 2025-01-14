import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Folder containing your files
# folder_path = "/scratch/alexis/data/2024-1205-pybullet-finetuning/iql-training-p2-reward_model1-only-12_05_05_50_38-000/gifs"
folder_path = "/scratch/alexis/data/2024-1205-pybullet-finetuning/iql-training-p1-reward_model1-only-12_05_07_16_53-000/gifs"

# Generalized pattern to extract motion, step, and last value
filename_pattern = re.compile(r".*_motion(\d+)_.*_step(\d+)_\d+\.\d+_(\d+\.\d+)\.gif$")
# Initialize a dictionary to hold data for each motion and step
data = {}

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    match = filename_pattern.match(filename)
    if match:
        motion, step, value = int(match[1]), int(match[2]), float(match[3])
        if motion not in data:
            data[motion] = {}
        if step not in data[motion]:
            data[motion][step] = []
        data[motion][step].append(value)

# Prepare data for plotting
for motion, steps in sorted(data.items()):
    steps = dict(sorted(steps.items()))  # Sort steps numerically
    x = []
    y = []

    for step, values in steps.items():
        x.append(step)
        y.append(np.mean(values))  # Compute the average of values

    # Plot the results for the current motion
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title(f"Motion {motion}: Average Upper Arm Dressed Ratio vs Step")
    plt.xlabel("Step")
    plt.ylabel("Average Upper Arm Dressed Ratio")
    plt.grid(True)
    plt.savefig(f"motion_{motion}_plot.png")  # Save the plot as an image
    plt.show()
