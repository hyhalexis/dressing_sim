import open3d as o3d
import torch
import torchvision  # Must import this for the model to load without error
import numpy as np
import matplotlib.pyplot as plt

import pickle5 as pickle
import os

model = torch.jit.load('models/nlf_l_multi.torchscript').cuda().eval()

input_file = '/home/alexis/assistive-gym-fem/assistive_gym/envs/temp'
with open(input_file, 'rb') as f:
    imgs = pickle.load(f)


image = torch.from_numpy(imgs[1]).permute(2, 0, 1).float().cuda()  # Shape: (C, H, W)
frame_batch = image.unsqueeze(0)

with torch.inference_mode(), torch.device('cuda'):
   pred = model.detect_smpl_batched(frame_batch)

# SMPL Parametric predictions
pred['pose'], pred['betas'], pred['trans']
pred['joints3d'], pred['vertices3d']
pred['joints2d'], pred['vertices2d']

# Nonparametric joints and vertices
pred['joints3d_nonparam'], pred['vertices3d_nonparam']
pred['joints2d_nonparam'], pred['vertices2d_nonparam']
pred['joint_uncertainties'], pred['vertex_uncertainties']

vertices_list = pred['vertices3d_nonparam']

# Convert list to tensor and get desired shape
vertices_tensor = torch.stack(vertices_list)  # Shape: (1, 2, 1024, 3)
vertices_tensor = vertices_tensor.squeeze(0)[0]  # Remove batch dim and select first set: (1024, 3)

# Convert to NumPy
vertices_numpy = vertices_tensor.cpu().detach().numpy()
import matplotlib.pyplot as plt

# Project onto the XY plane (ignore Z-axis)
import matplotlib.pyplot as plt
import numpy as np

# Assuming vertices_numpy contains the 3D coordinates (1024, 3)
x = vertices_numpy[:, 0]
y = vertices_numpy[:, 1]
z = vertices_numpy[:, 2]
y = -y  # Flip Y-axis if needed

# Compute the distance of each point from the origin (0, 0, 0)
distances = np.sqrt(x**2 + y**2 + z**2)

# Create a colormap based on distances
plt.figure(figsize=(8, 8))

# Increase the point size and adjust the alpha for better visibility
sc = plt.scatter(x, y, s=5, c=distances, cmap='plasma', alpha=1.0)

# Add a color bar to show the mapping from distance to color
#plt.colorbar(sc, label='Distance from Origin')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Estimated Pose")

# Set equal aspect ratio for the plot to ensure the points are not distorted
plt.axis('equal')

# Save the plot as an image
plt.savefig("traj_pose.jpg")