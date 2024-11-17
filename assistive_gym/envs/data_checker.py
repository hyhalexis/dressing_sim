import numpy as np
import pickle5 as pickle
import matplotlib.pyplot as plt
import open3d as o3d

traj_path = 'traj_data/p2_motion7_side_tshirt_68.pkl'
with open (traj_path, 'rb') as f:
    traj_data = pickle.load(f)

print('Length: {}'.format(len(traj_data)))
print('First entry: {}'.format(traj_data[0]))
print('Last entry: {}'.format(traj_data[-1]))
print(traj_data[0]['obs'].pos.shape)

image0 = traj_data[0]['img']
plt.imshow(image0)
# plt.show()
print('gripper0', traj_data[0]['gripper_pos'])
print('line0', traj_data[0]['line_points'])


image53 = traj_data[53]['img']
plt.imshow(image53)
# plt.show()
print('gripper53', traj_data[53]['gripper_pos'])
print('line53', traj_data[53]['line_points'])

def create_axes(length=0.5):
    # Create lines for the axes
    lines = [
        [0, 1],  # X axis
        [0, 2],  # Y axis
        [0, 3]   # Z axis
    ]

    # Create points for the axes
    points = [
        [0, 0, 0],  # Origin
        [length, 0, 0],  # X axis
        [0, length, 0],  # Y axis
        [0, 0, length]   # Z axis
    ]

    # Create LineSet
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)

    # Set colors for the axes
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red, Green, Blue
    lineset.colors = o3d.utility.Vector3dVector(colors)

    return lineset

# Create axes and visualize
axes = create_axes(length=0.5)


pc0 = traj_data[0]['obs'].pos.reshape(-1,3)
gripper0 = np.array(traj_data[0]['gripper_pos']).reshape(-1,3)
line0 = np.array(traj_data[0]['line_points']).reshape(-1,3)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pc0)

point_cloud2 = o3d.geometry.PointCloud()
point_cloud2.points = o3d.utility.Vector3dVector(gripper0)

point_cloud3 = o3d.geometry.PointCloud()
point_cloud3.points = o3d.utility.Vector3dVector(line0)



combined_pcd = point_cloud + point_cloud2 + point_cloud3
# combined_pcd = point_cloud
point_cloud.paint_uniform_color([1, 0, 0])
point_cloud2.paint_uniform_color([0, 0, 1])
o3d.visualization.draw_geometries([combined_pcd, axes],
                        window_name="Combined Point Clouds",
                        width=800,
                        height=600,
                        left=50,
                        top=50,
                        mesh_show_back_face=True)



