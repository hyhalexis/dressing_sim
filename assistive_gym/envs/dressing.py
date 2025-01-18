import os, time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import torch

from env import AssistiveEnv
from agents.human_mesh import HumanMesh
from garment_idx_utils import *
from PIL import Image
import open3d as o3d
from torch_geometric.data import Data
from utils import voxelize_pc, show_line_and_triangle, line_intersecting_triangle, filter_points, to_model_axes, distort_arm_pc, create_axes
import imageio
import scipy
import pickle
from scipy.spatial.transform import Rotation as R

import copy

class DressingEnv(AssistiveEnv):
    def __init__(self, robot, human, use_ik=True, policy=2, horizon=150, camera_pos = 'side', occlusion = True, render=False, one_hot = False, reconstruct = False, gif_path=None, use_force=False, elbow_rand=-90, shoulder_rand=80):
    # def __init__(self, robot, human, use_ik=True, policy=2, horizon=150, motion=1, garment=1, camera_pos = 'side', render=False):
        super(DressingEnv, self).__init__(robot=robot, human=human, task='dressing', obs_robot_len=(16 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(16 + (len(human.controllable_joint_indices) if human is not None else 0)), frame_skip=1, time_step=0.02, deformable=True, render=render)
        self.use_ik = use_ik
        self.use_mesh = (human is None)
        self.arm_traj_idx = 0
        self.arm_traj = None
        self.traj_direction = 1
        self.repeat_traj = 0
        # self.motion_id = int(motion)
        # self.garment_id = int(garment)
        self.verbose = False
        self.horizon = horizon
        self.policy = int(policy)
        self.camera_pos = camera_pos

        self.upper_w = 5
        self.task_w = 1
        self.force_w =0.001
        self.collision_w = 0.01
        self.force_threshold = 1000
        self.collision_threshold = 0.02
        self.particle_radius = 0.00625
        self.camera_width, self.camera_height = 360, 360
        self.cloth_forces = np.zeros(1)
        self.cloth_force_vector = np.zeros((1, 3))

        self.reward_image = None
        self.cap_reward = True
        self.near_center_range = 0.03
        self.far_center_range = 0.075
        self.center_align_reward_w = 0.05
        self.center_align_penalty_w = 0.02

        self.elastic_stiffness = 0.5  #0.5
        self.damping_stiffness = 0.01
        self.bending_stiffnes = 0.1 #0.1
        self.all_direction = 0
        self.useNeoHookean = 1

        self.step_img = None
        self.step_gripper_pos = None
        self.step_line_pts = []

        # self.arm_points = np.zeros((1,3))
        # self.collision_kd_tree = scipy.spatial.KDTree(self.arm_points)

        self.line1_extend_factor = 0.1

        self.images = []
        self.force_lst = []
        self.gif_path = gif_path

        self.poses_lst = []
        self.occlusion = occlusion
        self.img_with_force = []
        self.error_lst = [[],[]]
        self.use_force = use_force
        self.one_hot = one_hot
        self.reconstruct = reconstruct

        self.elbow_rand = elbow_rand
        self.shoulder_rand = shoulder_rand


        # 1 skip, 0.1 step, 50 substep, springElasticStiffness=100 # ? FPS
        # 1 skip, 0.1 step, 20 substep, springElasticStiffness=10 # 4.75 FPS
        # 5 skip, 0.02 step, 4 substep, springElasticStiffness=5 # 5.75 FPS
        # 10 skip, 0.01 step, 2 substep, springElasticStiffness=5 # 4.5 FPS

    def compute_reward(self):
        # Get cloth data
        x, y, z, cx, cy, cz, fx, fy, fz = p.getSoftBodyData(self.cloth, physicsClientId=self.id)
        mesh_points = np.concatenate([np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), np.expand_dims(z, axis=-1)], axis=-1)
        forces = np.concatenate([np.expand_dims(fx, axis=-1), np.expand_dims(fy, axis=-1), np.expand_dims(fz, axis=-1)], axis=-1) * 10
        contact_positions = np.concatenate([np.expand_dims(cx, axis=-1), np.expand_dims(cy, axis=-1), np.expand_dims(cz, axis=-1)], axis=-1)
        
        end_effector_pos, end_effector_orient = self.robot.get_pos_orient(self.robot.left_end_effector)
        forces_temp = []
        contact_positions_temp = []
        for f, c in zip(forces, contact_positions):
            # if c[-1] < end_effector_pos[-1] - 0.05 and np.linalg.norm(f) < 20:
            forces_temp.append(f)
            contact_positions_temp.append(c)
        self.cloth_forces = np.array(forces_temp)
        self.cloth_force_vector = np.sum(self.cloth_forces, axis=0)
        if forces_temp == []:
            self.cloth_force_vector = np.zeros((1, 3))
        else:
            if self.cloth_force_vector.shape[0] == 1:
                print('WTFFFFFFF', self.cloth_force_vector)
            self.cloth_force_vector = self.cloth_force_vector.reshape(-1, 3)

        cloth_shoulder_polygon_particle_pos = mesh_points[self.shoulder_polygon_indices]

        shoulder_center = np.mean(cloth_shoulder_polygon_particle_pos, axis=0) 
        reward_start_pos = self.line_points[0]
        # self.create_sphere(radius=0.05, mass=0.0, pos=shoulder_center, visual=True, collision=False, rgba=[0, 1, 1, 0.5], maximal_coordinates=False, return_collision_visual=False)
        
        for i in range(5):
            p.addUserDebugLine(cloth_shoulder_polygon_particle_pos[i], cloth_shoulder_polygon_particle_pos[i+1], lineWidth=5, lineColorRGB=[0, 1, 1], lifeTime=2.0)

        line_points = self.line_points
        line1_ori, line2_ori, line1_dir, line2_dir = self.line1_ori, self.line2_ori, self.line1_dir, self.line2_dir

        forward_distance = - np.linalg.norm(reward_start_pos - shoulder_center)

        # check if on forearm
        string = None
        on_forearm = False
        self.forearm_distance = 0
        self.upperarm_distance = 0
        self.intersect_point1 = None
        self.intersect_point2 = None
        for triangle_idx in self.reward_triangle_idxes:
            intersect_line1, intersect_point1 = line_intersecting_triangle(
                line1_ori - line1_dir * self.line1_extend_factor, 
                line1_dir, 
                mesh_points[triangle_idx[0]], mesh_points[triangle_idx[1]], mesh_points[triangle_idx[2]])
            if intersect_line1:
                vec = intersect_point1 - reward_start_pos
                on_fore_arm_sign = np.dot(vec, -line1_dir)
                if on_fore_arm_sign >= 0: # cloth pulled on forearm
                    self.intersect_point1 = intersect_point1
                    string = "sleeve enter forearm"
                    forward_distance = np.linalg.norm(vec)
                    # NOTE yufei: cap the forearm forward distance to be the lenght of forearm.
                    if self.cap_reward:
                        forward_distance = min(forward_distance, np.linalg.norm(line_points[1] - line_points[0]))
                    on_forearm = True
                    self.forearm_distance = forward_distance
                    break
                else:
                    string = "intersecting with forearm, but not really entered it!"
                print(string)
        self.on_forearm = on_forearm

        # check if on upper arm
        on_upperarm = False
        upper_arm_distances = []
        intersection_points = []
        for triangle_idx in self.reward_triangle_idxes:
            intersect_line2, intersect_point2 = line_intersecting_triangle(line2_ori - line2_dir * 0.1, line2_dir, 
                mesh_points[triangle_idx[0]], mesh_points[triangle_idx[1]], mesh_points[triangle_idx[2]])
            if intersect_line2:
                vec2 = intersect_point2 - line1_ori
                on_upper_arm_sign = np.dot(vec2, -line2_dir)
                if on_upper_arm_sign >= 0: # cloth pulled on upperarm
                    string = "sleeve enter upperarm"
                    on_upperarm = True
                    self.forearm_distance = np.linalg.norm(line_points[1] - line_points[0]) + self.human.hand_radius
                    forward_distance = self.forearm_distance + \
                        self.upper_w * np.linalg.norm(vec2)
                    self.upperarm_distance = np.linalg.norm(vec2)
                    upper_arm_distances.append(np.linalg.norm(vec2))
                    intersection_points.append(intersect_point2)
                else:
                    string = "intersecting with upper arm, but not really entered it!"
                print(string)

        if on_upperarm:
            min_distance = min(upper_arm_distances)
            forward_distance = np.linalg.norm(line_points[1] - line_points[0]) + self.upper_w * min_distance
            self.upperarm_distance = min_distance
            self.intersect_point2 = intersection_points[np.argmin(upper_arm_distances)]

        self.on_upperarm = on_upperarm
        self.forward_distance = forward_distance

        stay_close_to_intersection_reward = 0
        intersection = None
        intersection = self.intersect_point2 if on_upperarm else self.intersect_point1
        center_to_intersect_distance = 0

        if on_upperarm:
        # if intersection is not None:
            center_to_intersect_distance = np.linalg.norm(shoulder_center - intersection)
            if center_to_intersect_distance < self.near_center_range:
                stay_close_to_intersection_reward = 1 * self.center_align_reward_w
            if center_to_intersect_distance > self.far_center_range:
                stay_close_to_intersection_reward = -1 * self.center_align_penalty_w
        self.stay_close_to_intersection_reward = stay_close_to_intersection_reward
        self.center_to_intersect_distance = center_to_intersect_distance

        gripper_pos = end_effector_pos.reshape(1, 3)

        if string is None:
            string = "sleeve not entered arm yet"

        grasped_particles = mesh_points[self.anchor_vertices]

        self.reward_image = show_line_and_triangle(self.forearm_distance, self.upperarm_distance,
                self.camera_width, self.camera_height, 
                self.voxelized_observable_cloth_pc,
                grasped_particles,
                self.voxelized_observable_arm_pc,
                cloth_shoulder_polygon_particle_pos, 
                line1_ori - line1_dir * self.line1_extend_factor, line1_dir, line2_ori, line2_dir,
                reward_start_pos, end_effector_pos, 
                self.intersect_point1, self.intersect_point2,
                False)
 
        task_reward = max(-20, forward_distance)


        # # Get 3D points for triangles around the sleeve to detect if the sleeve is around the arm
        # triangle1_points = mesh_points[self.triangle1_point_indices]
        # triangle2_points = mesh_points[self.triangle2_point_indices]
        # triangle3_points = mesh_points[self.triangle3_point_indices]
        # triangle4_points = mesh_points[self.triangle4_point_indices]

        # # TODO: fix to make it work with four triangles
        # self.forearm_in_sleeve, self.upperarm_in_sleeve, self.distance_along_forearm, self.distance_along_upperarm, self.distance_to_hand, self.distance_to_elbow, self.distance_to_shoulder, self.forearm_length, self.upperarm_length = self.util.sleeve_on_arm_reward(triangle1_points, triangle2_points, triangle3_points, triangle4_points, shoulder_pos, elbow_pos, wrist_pos, self.human.hand_radius, self.human.elbow_radius, self.human.shoulder_radius)

        # if self.upperarm_in_sleeve:
        #     task_reward = self.forearm_length
        #     if self.distance_along_upperarm < self.upperarm_length:
        #         task_reward += self.distance_along_upperarm * self.upper_w
        # elif self.forearm_in_sleeve and self.distance_along_forearm < self.forearm_length:
        #     task_reward = self.distance_along_forearm
        # else:
        #     task_reward = -self.distance_to_hand


        if self.total_force_on_human > self.force_threshold:
            force_reward = -(self.total_force_on_human - self.force_threshold)
        else:
            force_reward = 0

        self.collision_penalty = 0
        if self.closest_dist is not None and self.closest_dist < self.particle_radius + self.collision_threshold: 
            self.collision_penalty = -1

        reward = self.task_w * task_reward + \
            self.force_w * force_reward + \
            self.collision_w * self.collision_penalty
        if self.center_align_reward_w > 0:
            reward += stay_close_to_intersection_reward
       
        reward = max(reward, -100)
        self.reward = reward
        if reward == -100:
            print("reward lower bound simulation error!")
        return reward

    def step(self, action):
        # if self.human.controllable:
        #     action = np.concatenate([action['robot'], action['human']])
        # action = np.ones(7)

        end_effector_pos, end_effector_orient = self.robot.get_pos_orient(self.robot.left_end_effector)
        new_ee_pos = end_effector_pos + action[:3]
        # self.create_sphere(radius=0.01, mass=0.0, pos=end_effector_pos, visual=True, collision=False, rgba=[0, 1, 1, 0.5], maximal_coordinates=False, return_collision_visual=False)
        # self.create_sphere(radius=0.01, mass=0.0, pos=new_ee_pos, visual=True, collision=False, rgba=[1, 0, 0, 0.5], maximal_coordinates=False, return_collision_visual=False)
        
        self.collision_kd_tree = scipy.spatial.KDTree(self.arm_points)

        # p.resetBasePositionAndOrientation(self.robot, new_ee_pos, end_effector_orient)

        # collision_points = p.getClosestPoints(bodyA=self.robot, bodyB=self.human, distance=0.02)
    
        min_distances_kd_tree, _ = self.collision_kd_tree.query(new_ee_pos, workers=8)
        # min_distances_kd_tree_ori, _ = self.collision_kd_tree.query(end_effector_pos, workers=8)

        min_distance = np.min(min_distances_kd_tree)
        # print('distance to arm', min_distances_kd_tree_ori)

        # min_distance = self.robot.get_closest_points(self.human, distance=5.0)[-2]
        # print('next distance to arm', np.linalg.norm(min_distance))
        # next_min_distance = min_distance + action[:3]
        # print('new distance to arm', np.linalg.norm(next_min_distance))
        # if min_distance < self.collision_threshold:
        #     print('skipped!')
        #     # import pdb; pdb.set_trace()
        # else:
        if self.use_ik:
            self.take_step(action, ik=True)
        else:
            self.take_step(action)

        obs, force_vector = self._get_obs()
        reward = self.compute_reward()
        info = self._get_info()
        done = self.iteration >= self.horizon or info['upperarm_ratio'] > 0.99

        if done:
            # plt.figure(1)
            # plt.plot(self.error_lst[0])
            # plt.title('IK Error close_multi')
            # plt.savefig('sim_imgs/ik_close_multi.png')

            # # Create the second figure
            # plt.figure(2)
            # plt.plot(self.error_lst[1])
            # plt.title('Execution Error close_multi')
            # plt.savefig('sim_imgs/exec_close_multi.png')
            # plt.close('all')
            # print('view: ', self.view_matrix)
            # print('proj: ', self.projection_matrix)
   
            # imageio.mimsave('sim_gifs/up_p{}_motion{}_{}_{}_step{}_{}_{}.gif'.format(self.policy, self.motion_id, self.camera_pos, self.garment, self.step_idx, (self.upperarm_distance + self.forearm_distance) / (self.upper_arm_length + self.forearm_length), self.upperarm_distance / self.upper_arm_length), self.img_with_force, format='GIF', duration=30)
            # imageio.mimsave('{}/pose{}_{}_{}_step{}_{}_{}.gif'.format(self.gif_path, self.pose_id, self.camera_pos, self.garment, self.step_idx, (self.upperarm_distance + self.forearm_distance) / (self.upper_arm_length + self.forearm_length), self.upperarm_distance / self.upper_arm_length), self.images, format='GIF', duration=30)

            imageio.mimsave('{}/p{}_motion{}_{}_{}_step{}_e{}_s{}_force{}_{}_{}.gif'.format(self.gif_path, self.policy, self.motion_id, self.camera_pos, self.garment, self.step_idx, self.elbow_rand, self.shoulder_rand, self.use_force, (self.upperarm_distance + self.forearm_distance) / (self.upper_arm_length + self.forearm_length), self.upperarm_distance / self.upper_arm_length), self.img_with_force, format='GIF', duration=30)
            # imageio.mimsave('{}/p{}_motion{}_{}_{}_e{}_s{}_{}_{}.gif'.format(self.gif_path, self.policy, self.motion_id, self.camera_pos, self.garment, self.elbow_rand, self.shoulder_rand, (self.upperarm_distance + self.forearm_distance) / (self.upper_arm_length + self.forearm_length), self.upperarm_distance / self.upper_arm_length), self.images, format='GIF', duration=30)
            # imageio.mimsave('sim_gifs/test_reduced_pts_motion{}_pose{}.gif'.format(self.motion_id, self.pose_id), self.images, format='GIF', duration=30)
        # print('------Iteration', self.iteration)
        print(info)
        # if self.gui:
        #     print('Task success:', self.task_success, 'Dressing reward:', reward_dressing)
        return obs, reward, done, info, force_vector

    def _get_info(self):
        ret_dict = {}
        ret_dict['task_reward'] = self.reward
        # ret_dict['force_reward'] = self.force_reward * self.force_w
        ret_dict['forearm_in_sleeve'] = int(self.on_forearm)
        ret_dict['upperarm_in_sleeve'] = int(self.on_upperarm)
        ret_dict['forward_distance'] = self.forward_distance
        ret_dict['forearm_distance'] = self.forearm_distance
        ret_dict['upperarm_distance'] = self.upperarm_distance
        ret_dict['upperarm_ratio'] = self.upperarm_distance / self.upper_arm_length
        ret_dict['whole_arm_ratio'] = (self.upperarm_distance + self.forearm_distance) / (self.upper_arm_length + self.forearm_length)
        ret_dict['collision_penalty'] = self.collision_penalty * self.collision_w
        ret_dict['gripper closes dist'] = self.closest_dist
        ret_dict['force on human'] = self.total_force_on_human
        # ret_dict['stay_close_to_intersection_reward'] = self.stay_close_to_intersection_reward
        # ret_dict['center_to_intersect_distance'] = self.center_to_intersect_distance
        ret_dict['reward'] = self.reward
        return ret_dict

    def _get_obs(self):
        end_effector_pos, end_effector_orient = self.robot.get_pos_orient(self.robot.left_end_effector)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        shoulder_pos = self.human.get_pos_orient(self.human.j_right_shoulder_x)[0]
        elbow_pos = self.human.get_pos_orient(self.human.j_right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_hand)[0]
        self.cloth_force_sum = np.sum(np.linalg.norm(self.cloth_forces, axis=-1))
        self.robot_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        self.total_force_on_human = self.robot_force_on_human + self.cloth_force_sum
        self.force_lst.append(self.total_force_on_human)
        self.closest_dist = min(self.robot.get_closest_points(self.human, 0.02)[-1]) if self.robot.get_closest_points(self.human, 0.02)[-1] else None

        line_points = [wrist_pos+[0, -self.human.hand_radius, 0], elbow_pos, shoulder_pos+ [-self.human.shoulder_radius, 0, 0]]
        # line_points = [wrist_pos+[self.human.hand_radius, -self.human.hand_radius, 0], elbow_pos+[self.human.hand_radius, -self.human.hand_radius, 0], shoulder_pos+[0, -self.human.hand_radius, 0]]

        line1 = line_points[0] - line_points[1] # from elbow to finger
        line2 = line_points[1] - line_points[2] # from shoulder to elbow
        # p.addUserDebugLine(line_points[0], line_points[1], lineColorRGB=[1, 0, 0], lifeTime=2.0)  # Red line
        # p.addUserDebugLine(line_points[2], line_points[1], lineColorRGB=[0, 1, 0], lifeTime=2.0)  # Red line

        self.upper_arm_length = np.linalg.norm(line2)
        # print('----Upper Arm Length', self.upper_arm_length)
        self.forearm_length = np.linalg.norm(line1) + self.human.hand_radius
        self.line1_ori, self.line2_ori = line_points[1], line_points[2] # elbow, shoulder
        self.line1_dir, self.line2_dir = line1 / np.linalg.norm(line1), line2 / np.linalg.norm(line2)
        self.line_points = line_points

        self.step_gripper_pos = end_effector_pos
        self.step_line_pts = line_points
        
        # if self.camera_pos == 'front':
        #     self.setup_camera_rpy(camera_target=[-0.31471434, -0.2672464, 1.00759685], distance=0.7, rpy=[0, -25, 0], fov=60, camera_width=1080, camera_height=720)
        # elif self.camera_pos == 'side':
        #     self.setup_camera_rpy(camera_target=[-0.01471434, -0.5672464, 1.00759685], distance=1.0, rpy=[0, -20, 15], fov=60, camera_width=1080, camera_height=720)

        # self.setup_camera_rpy(camera_target=[-0.01471434, -0.5672464, 1.00759685], distance=1.0, rpy=[0, -25, 0], fov=60, camera_width=1080, camera_height=720)
        
        # self.setup_camera_rpy(camera_target=[-0.01471434, -0.5672464, 1.00759685], distance=1.0, rpy=[0, -20, 15], fov=60, camera_width=1080, camera_height=720)

        img, _, _ = self.get_camera_image_depth()
        img = np.array(img)
        # plt.imshow(img)
        # plt.show()

        dpi = 100
        fig = plt.figure(figsize=(self.camera_width/dpi, self.camera_height/dpi), dpi=dpi)
        ax_total = fig.add_subplot(1, 1, 1) # plot total reward

        ax_total.plot(range(len(self.force_lst)), self.force_lst, label='total_reward', color='C0')
        ax_total.legend()

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close("all")

        self.images.append(img)
        self.img_with_force.append(np.concatenate([img[..., :3], image_from_plot], axis=1))
        # [-0.39448014, -1.20101663, 1.225]
        if not self.occlusion:
            p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 0], flags=0, physicsClientId=self.id)
            arm_points, arm_depth, arm_colors = self.get_point_cloud(self.human.body)
            whole_arm_points, whole_arm_depth = copy.deepcopy(arm_points), copy.deepcopy(arm_depth)
            p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 1], flags=0, physicsClientId=self.id)
            cloth_points, cloth_depth, cloth_colors = self.get_point_cloud(self.cloth)
        else:
            arm_points, arm_depth, arm_colors = self.get_point_cloud(self.human.body)
            cloth_points, cloth_depth, cloth_colors = self.get_point_cloud(self.cloth)

            p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 0], flags=0, physicsClientId=self.id)
            whole_arm_points, whole_arm_depth, arm_colors = self.get_point_cloud(self.human.body)
            p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 1], flags=0, physicsClientId=self.id)

        observable_cloth_pc = cloth_points[cloth_depth > 0].astype(np.float32)
        average_arm_height = np.mean(arm_points, axis=0)[2]
        cloth_height_threshold = average_arm_height - 0.3
        observable_cloth_pc = observable_cloth_pc[observable_cloth_pc[:, 2] > cloth_height_threshold]
        # import pdb; pdb.set_trace()
        
        # merged_pc = arm_points.copy()
        # zero_depth_indices = arm_points[:, 2] == 0
        # merged_pc[zero_depth_indices] = cloth_points[zero_depth_indices]
        # self.pc_images.append(merged_pc)


        self.arm_points = arm_points
        # if self.iteration == 1:
        # self.collision_kd_tree = scipy.spatial.KDTree(self.arm_points)
        
        if self.verbose:
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

            # Check if points are valid
            if arm_points is not None and len(arm_points) > 0:
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(arm_points)
                # point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
                o3d.visualization.draw_geometries([point_cloud, axes])
            
            if cloth_points is not None and len(cloth_points) > 0:
                point_cloud2 = o3d.geometry.PointCloud()
                point_cloud2.points = o3d.utility.Vector3dVector(cloth_points)
                # point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
                o3d.visualization.draw_geometries([point_cloud2, axes])

            combined_pcd = point_cloud + point_cloud2
            point_cloud.paint_uniform_color([1, 0, 0])
            point_cloud2.paint_uniform_color([0, 0, 1])
            o3d.visualization.draw_geometries([combined_pcd, axes],
                                    window_name="Combined Point Clouds",
                                    width=800,
                                    height=600,
                                    left=50,
                                    top=50,
                                    mesh_show_back_face=True)
            
            # image = Image.fromarray(img[..., :3], 'RGB')
            # image.save('./captured_image.png')
            # plt.imshow(depth, cmap='gray')
            # print('saved')
            # # plt.imshow(image)
            # plt.show()

        # Separate the points into x, y, z components for plotting
        # x_vals = points[:, 0]
        # y_vals = points[:, 1]
        # z_vals = points[:, 2]

        # # Create a 3D scatter plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis', marker='o')  # Color by depth

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.title('3D Point Cloud')
        # print('made!!')
        # plt.show()

        # temp = arm_depth.reshape((360,360))
        # plt.imshow(temp, cmap='gray')
        # plt.colorbar()  # Optional: Add a color bar to indicate depth values
        # plt.title('Depth Image')
        # plt.axis('off')  # Turn off axis
        # plt.show()

        # print('depth shape', cloth_depth.shape, arm_depth.shape)
        
        observable_cloth_pc = observable_cloth_pc[:, [1, 2, 0]]
        # observable_cloth_pc = cloth_points

        observable_arm_pc= arm_points[arm_depth > 0].astype(np.float32)
        observable_arm_pc = observable_arm_pc[:, [1, 2, 0]]

        whole_arm_pc= whole_arm_points[whole_arm_depth > 0].astype(np.float32)
        whole_arm_pc = whole_arm_pc[:, [1, 2, 0]]

        # observable_arm_pc = arm_points

        self.voxelized_observable_arm_pc = voxelize_pc(observable_arm_pc, 0.00625 * 10)
        self.voxelized_observable_cloth_pc = voxelize_pc(observable_cloth_pc, 0.00625 * 10)
        self.voxelized_whole_arm_pc = voxelize_pc(whole_arm_pc, 0.00625 * 10)

        arm_joint_points = [to_model_axes(line_points[0]), to_model_axes(line_points[1]), to_model_axes(line_points[2])]
        self.distort_arm_pc = distort_arm_pc(self.voxelized_whole_arm_pc, arm_joint_points)
        # print('before', self.voxelized_observable_arm_pc)

        # arm_lines = [(to_model_axes(line_points[0]+[0, 0, self.human.shoulder_radius]), to_model_axes(line_points[1]+[0, 0, self.human.shoulder_radius])), (to_model_axes(line_points[1]+[0, 0, self.human.shoulder_radius]), to_model_axes(line_points[2]+[0, 0, self.human.shoulder_radius]))]

        # self.voxelized_observable_arm_pc = filter_points(self.voxelized_observable_arm_pc, arm_lines, threshold=0.05)
        # print('after', self.voxelized_observable_arm_pc)

        end_effector_pos = end_effector_pos.reshape(-1, 3)
        end_effector_pos = end_effector_pos[:, [1 ,2 ,0]]

        if self.verbose:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            create_axes(ax, length=0.5)

            ax.scatter(self.distort_arm_pc[:, 0], self.distort_arm_pc[:, 1], self.distort_arm_pc[:, 2], c='k', s=10)  # Point color is black, size 10
            ax.scatter(self.voxelized_observable_arm_pc[:, 0], self.voxelized_observable_arm_pc[:, 1], self.voxelized_observable_arm_pc[:, 2], c='b', s=10)  # Point color is black, size 10
            plt.show()

        if self.occlusion:
            if self.use_force:
                if self.one_hot:
                    if self.reconstruct:
                        partial_pc = np.concatenate([self.voxelized_observable_arm_pc, self.voxelized_observable_cloth_pc], axis=0)
                        scene_pc = np.concatenate([partial_pc, self.distort_arm_pc], axis=0)
                        all_points = np.concatenate([scene_pc, end_effector_pos], axis=0)
                        all_points = all_points - end_effector_pos

                        categories = np.zeros((all_points.shape[0], 5))
                        categories[:len(self.voxelized_observable_arm_pc), 0] = 1
                        categories[len(self.voxelized_observable_arm_pc):len(self.voxelized_observable_arm_pc) + len(self.voxelized_observable_cloth_pc), 1] = 1
                        categories[partial_pc.shape[0]:scene_pc.shape[0], 2] = 1
                        categories[scene_pc.shape[0]:, 3] = 1
                        categories[scene_pc.shape[0]:, 4] = self.total_force_on_human
                        data = Data(pos=torch.from_numpy(all_points).float(), x=torch.from_numpy(categories).float())
                    else:
                        partial_pc = np.concatenate([self.voxelized_observable_arm_pc, self.voxelized_observable_cloth_pc], axis=0)
                        all_points = np.concatenate([partial_pc, end_effector_pos], axis=0)
                        all_points = all_points - end_effector_pos

                        categories = np.zeros((all_points.shape[0], 4))
                        categories[:len(self.voxelized_observable_arm_pc), 0] = 1 # arm
                        categories[len(self.voxelized_observable_arm_pc):len(self.voxelized_observable_arm_pc) + len(self.voxelized_observable_cloth_pc), 1] = 1 # cloth
                        categories[len(self.voxelized_observable_arm_pc) + len(self.voxelized_observable_cloth_pc):, 2] = 1 # gripper
                        categories[partial_pc.shape[0]:, 3] = self.total_force_on_human
                        data = Data(pos=torch.from_numpy(all_points).float(), x=torch.from_numpy(categories).float())

                elif self.reconstruct:
                    partial_pc = np.concatenate([self.voxelized_observable_arm_pc, self.voxelized_observable_cloth_pc], axis=0)
                    scene_pc = np.concatenate([partial_pc, self.distort_arm_pc], axis=0)
                    all_points = np.concatenate([scene_pc, end_effector_pos], axis=0)
                    all_points = all_points - end_effector_pos

                    categories = np.zeros((all_points.shape[0], 4))
                    categories[:partial_pc.shape[0], 0] = 1
                    categories[partial_pc.shape[0]:scene_pc.shape[0], 1] = 1
                    categories[scene_pc.shape[0]:, 2] = 1
                    categories[scene_pc.shape[0]:, 3] = self.total_force_on_human
                    data = Data(pos=torch.from_numpy(all_points).float(), x=torch.from_numpy(categories).float())

                else:
                    partial_pc = np.concatenate([self.voxelized_observable_arm_pc, self.voxelized_observable_cloth_pc], axis=0)
                    all_points = np.concatenate([partial_pc, end_effector_pos], axis=0)
                    all_points = all_points - end_effector_pos

                    categories = np.zeros((all_points.shape[0], 3))
                    categories[:partial_pc.shape[0], 0] = 1
                    categories[partial_pc.shape[0]:, 1] = 1
                    categories[partial_pc.shape[0]:, 2] = self.total_force_on_human
                    data = Data(pos=torch.from_numpy(all_points).float(), x=torch.from_numpy(categories).float())

            else:
                if self.one_hot:
                    partial_pc = np.concatenate([self.voxelized_observable_arm_pc, self.voxelized_observable_cloth_pc], axis=0)
                    all_points = np.concatenate([partial_pc, end_effector_pos], axis=0)
                    all_points = all_points - end_effector_pos
                    categories = np.zeros((all_points.shape[0], 3))
                    categories[:len(self.voxelized_observable_arm_pc), 0] = 1 # arm
                    categories[len(self.voxelized_observable_arm_pc):len(self.voxelized_observable_arm_pc) + len(self.voxelized_observable_cloth_pc), 1] = 1 # cloth
                    categories[len(self.voxelized_observable_arm_pc) + len(self.voxelized_observable_cloth_pc):, 2] = 1 # gripper
                    data = Data(pos=torch.from_numpy(all_points).float(), x=torch.from_numpy(categories).float())

                    scene_pc = np.concatenate([partial_pc, self.distort_arm_pc], axis=0)
                    rec_points = np.concatenate([scene_pc, end_effector_pos], axis=0)
                    rec_points = rec_points - end_effector_pos
                    rec_categories = np.zeros((rec_points.shape[0], 5))
                    rec_categories[:len(self.voxelized_observable_arm_pc), 0] = 1
                    rec_categories[len(self.voxelized_observable_arm_pc):len(self.voxelized_observable_arm_pc) + len(self.voxelized_observable_cloth_pc), 1] = 1
                    rec_categories[partial_pc.shape[0]:scene_pc.shape[0], 2] = 1
                    rec_categories[scene_pc.shape[0]:, 3] = 1
                    rec_categories[scene_pc.shape[0]:, 4] = self.total_force_on_human
                    self.complete_data = Data(pos=torch.from_numpy(rec_points).float(), x=torch.from_numpy(rec_categories).float())

                else:
                    partial_pc = np.concatenate([self.voxelized_observable_arm_pc, self.voxelized_observable_cloth_pc], axis=0)
                    all_points = np.concatenate([partial_pc, end_effector_pos], axis=0)
                    all_points = all_points - end_effector_pos
                    categories = np.zeros((all_points.shape[0], 2))
                    categories[:partial_pc.shape[0], 0] = 1
                    categories[partial_pc.shape[0]:, 1] = 1
                    data = Data(pos=torch.from_numpy(all_points).float(), x=torch.from_numpy(categories).float())

                    # scene_pc = np.concatenate([partial_pc, self.distort_arm_pc], axis=0)
                    # rec_points = np.concatenate([scene_pc, end_effector_pos], axis=0)
                    # rec_points = rec_points - end_effector_pos
                    # rec_categories = np.zeros((rec_points.shape[0], 4))
                    # rec_categories[:partial_pc.shape[0], 0] = 1
                    # rec_categories[partial_pc.shape[0]:scene_pc.shape[0], 1] = 1
                    # rec_categories[scene_pc.shape[0]:, 2] = 1
                    # rec_categories[scene_pc.shape[0]:, 3] = self.total_force_on_human
                    # self.complete_data = Data(pos=torch.from_numpy(rec_points).float(), x=torch.from_numpy(rec_categories).float())
        else:
            pointcloud = np.concatenate([self.voxelized_observable_arm_pc, self.voxelized_observable_cloth_pc, end_effector_pos], axis=0)
            pointcloud = pointcloud - end_effector_pos
            categories = np.zeros((len(pointcloud), 3)).astype(np.float32)

            categories[:len(self.voxelized_observable_arm_pc), 0] = 1 # arm
            categories[len(self.voxelized_observable_arm_pc):len(self.voxelized_observable_arm_pc) + len(self.voxelized_observable_cloth_pc), 1] = 1 # cloth
            categories[len(self.voxelized_observable_arm_pc) + len(self.voxelized_observable_cloth_pc):, 2] = 1 # gripper

            data = Data(pos=torch.from_numpy(pointcloud).float(), x=torch.from_numpy(categories).float())

        if self.verbose:
            def create_axes(ax, length=0.5):
                # Create lines for the axes
                lines = [
                    [[0, 0, 0], [length, 0, 0]],  # X axis
                    [[0, 0, 0], [0, length, 0]],  # Y axis
                    [[0, 0, 0], [0, 0, length]]   # Z axis
                ]
                
                # Set colors for the axes
                colors = ['r', 'g', 'b']  # Red, Green, Blue

                # Plot the axes
                for line, color in zip(lines, colors):
                    ax.plot3D(*zip(*line), color=color)

            # Create figure and 3D axis
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Create and plot axes
            create_axes(ax, length=0.5)
            pc = data.pos.cpu().detach().numpy()

            # Plot the point cloud
            if pc is not None and len(pc) > 0:
                ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='k', s=10)  # Point color is black, size 10

            # Set labels
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')

            # Show the plot
            plt.show()
            plt.savefig('sim_imgs/pc')
        # print(data.pos)
        # from matplotlib import pyplot as plt
        # ax = plt.axes(projection='3d')
        # ax.scatter(self.voxelized_observable_arm_pc[:, 0], self.voxelized_observable_arm_pc[:, 1], self.voxelized_observable_arm_pc[:, 2], color='red')
        # ax.scatter(self.voxelized_observable_cloth_pc[:, 0], self.voxelized_observable_cloth_pc[:, 1], self.voxelized_observable_cloth_pc[:, 2], color='blue')
        # ax.set_xlabel('X Axis Label')
        # ax.set_ylabel('Y Axis Label')
        # ax.set_zlabel('Z Axis Label')
        # if self.iteration % 30 == 0:
            # plt.show()
        return data, torch.from_numpy(self.cloth_force_vector).float()

    def reset(self, garment_id=1, motion_id=0, pose_id=-1, step_idx=0):
        super(DressingEnv, self).reset()
        self.garment_id = int(garment_id)
        self.motion_id = int(motion_id)
        self.pose_id = int(pose_id)
        self.step_idx = step_idx
        self.images = []
        self.build_assistive_env('wheelchair_left', gender='female', human_impairment='none')
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, np.pi/2.0])

        # Update robot and human motor gains
        # self.human.motor_gains = 0.15
        self.human.motor_gains = 0.05
        self.human.motor_forces = 100.0

        self.robot.motor_gains = 0.05 # 0.1
        self.robot.motor_forces = 100.0

        # self.generate_init_poses()
        # init_pose = self.poses_lst[pose_id]

        if self.use_mesh:
            self.human = HumanMesh()
            joints_positions = [(self.human.j_right_shoulder_z, 60), (self.human.j_right_elbow_y, 90), (self.human.j_left_shoulder_z, -10), (self.human.j_left_elbow_y, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee_x, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee_x, 80)]
            body_shape = np.zeros((1, 10))
            gender = 'female' # 'random'
            self.human.init(self.directory, self.id, self.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions, left_hand_pose=[[-2, 0, 0, -2, 0, 0]])

            chair_seat_position = np.array([0, 0.1, 0.55])
            self.human.set_base_pos_orient(chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])
        else:
            rng = np.random.default_rng()
            # self.elbow_rand = -90 + rng.uniform(-10, 10)
            # self.shoulder_rand = 80 + rng.uniform(-5, 5)
            print(self.elbow_rand, self.shoulder_rand)
            joints_positions = [(self.human.j_right_elbow, self.elbow_rand), (self.human.j_right_shoulder_x, self.shoulder_rand), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
            # self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=1, reactive_gain=0.01)

            if self.pose_id == -1:
                self.human.set_joint_angles([j for j, _ in joints_positions], [np.deg2rad(j_angle) for _, j_angle in joints_positions])
            
            else:
                self.human.set_joint_angles([j for j, _ in joints_positions], [np.deg2rad(j_angle) for _, j_angle in joints_positions])
                self.generate_init_poses()
                init_pose = self.poses_lst[pose_id]
                joints = [self.human.j_right_pecs_x, self.human.j_right_pecs_y, self.human.j_right_pecs_z, self.human.j_right_shoulder_x, self.human.j_right_shoulder_y, self.human.j_right_shoulder_z, self.human.j_right_elbow, self.human.j_right_forearm, self.human.j_right_wrist_x, self.human.j_right_wrist_y]
                joint_angles = init_pose.tolist()
                self.human.set_joint_angles([j for j in joints], [j_angle for j_angle in joint_angles])

            self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), self.human.motor_gains, self.human.motor_forces)
            # self.human.control(self.human.controllable_joint_indices, init_pose, self.human.motor_gains, self.human.motor_forces)

        if self.camera_pos == 'front':
            self.setup_camera_rpy(camera_target=[-0.31471434, -0.2672464, 1.00759685], distance=0.7, rpy=[0, -25, 0], fov=60, camera_width=1080, camera_height=720)
        elif self.camera_pos == 'side':
            self.setup_camera_rpy(camera_target=[-0.01471434, -0.5672464, 1.00759685], distance=1.0, rpy=[0, -20, 15], fov=60, camera_width=1080, camera_height=720)


        # if self.human.controllable or self.human.impairment == 'tremor':
        #     self.human.control(self.human.controllable_joint_indices, self.human.get_joint_angles(self.human.controllable_joint_indices), 0.05, 1)

        shoulder_pos = self.human.get_pos_orient(self.human.j_right_shoulder_x)[0]
        elbow_pos = self.human.get_pos_orient(self.human.j_right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_hand)[0]

        line_points = [wrist_pos+[0, -self.human.hand_radius, 0], elbow_pos, shoulder_pos+ [-self.human.shoulder_radius, 0, 0]]
        # line_points = [wrist_pos+[self.human.hand_radius, -self.human.hand_radius, 0], elbow_pos+[self.human.hand_radius, -self.human.hand_radius, 0], shoulder_pos+[0, -self.human.hand_radius, 0]]
        line1 = line_points[0] - line_points[1] # from elbow to finger
        line2 = line_points[1] - line_points[2] # from shoulder to elbow
        # p.addUserDebugLine(line_points[0], line_points[1], lineColorRGB=[1, 0, 0], lifeTime=0)  # Red line
        # p.addUserDebugLine(line_points[2], line_points[1], lineColorRGB=[0, 1, 0], lifeTime=0)  # Red line

        self.upper_arm_length = np.linalg.norm(line2)
        self.forearm_length = np.linalg.norm(line1) + self.human.hand_radius
        self.line1_ori, self.line2_ori = line_points[1], line_points[2] # elbow, shoulder
        self.line1_dir, self.line2_dir = line1 / np.linalg.norm(line1), line2 / np.linalg.norm(line2)
        self.line_points = line_points

        # base_pos = [-1.17448014, -0.30101663, 0.825]
        base_pos = [-1.00448014, -0.30101663, 0.825]
        base_orient = [0., 0., 0., 1.]

        # base_orient = [0., 0., -0.25249673, 0.96759775]

        # hand_pos, hand_ori = self.human.get_pos_orient(18, False, True)
        # hand_pos_robot = self.robot.convert_to_base_frame(hand_pos)[0]
        # print(hand_pos_robot)
        # joint_angles = self.robot.ik(self.robot.left_end_effector, hand_pos_robot, None, self.robot.right_arm_ik_indices, max_iterations=1000, use_current_as_rest=False)
        
        # hand_pos, hand_ori = self.human.get_pos_orient(18, False, convert_to_realworld=True)
        # import pdb; pdb.set_trace()
        wrist_pos, hand_ori = self.human.get_pos_orient(18)
        hand_pos = []
        for i in range(len(wrist_pos)):
            hand_pos.append(wrist_pos[i] + 2 * self.human.hand_radius * self.line1_dir[i])

        # self.create_sphere(radius=0.05, mass=0.0, pos=hand_pos, visual=True, collision=False, rgba=[0, 1, 1, 0.5], maximal_coordinates=False, return_collision_visual=False)
        
        # hand_pos[1] -= 0.57
        # hand_pos[2] += 0.11

        hand_pos[1] -= 0.49
        hand_pos[2] += 0.11

        # self.create_sphere(radius=0.05, mass=0.0, pos=self.line_points[0], visual=True, collision=False, rgba=[1, 0, 0, 0.5], maximal_coordinates=False, return_collision_visual=False)

        # hand_pos[0] += 0.09
        # hand_pos[1] -= 1.3
        # hand_pos[2] += 0.925

        # hand_pos_robot, hand_ori_robot = self.robot.convert_to_base_frame(hand_pos, hand_ori)
        hand_euler = self.robot.get_euler([0., 0., 0., 1.])
        hand_euler[1] += np.pi

        self.robot.reset_joints()
        self.robot.set_base_pos_orient(base_pos, base_orient)
        # joint_angles = self.robot.ik(self.robot.right_end_effector, hand_pos, hand_euler, self.robot.right_arm_ik_indices, max_iterations=1000, use_current_as_rest=True)


        for _ in range(1):
            joint_angles = self.robot.ik(self.robot.right_end_effector, hand_pos, hand_euler, self.robot.right_arm_ik_indices, max_iterations=1000, use_current_as_rest=True)
            self.robot.set_joint_angles(self.robot.controllable_joint_indices, joint_angles)
            # ee, ori = self.robot.get_pos_orient(self.robot.right_end_effector)
            # print('robot ori', self.robot.get_euler(ori))
            # p.stepSimulation(physicsClientId=self.id)
        
        # joint_angles = [-1.64317203, -0.95115511, -1.96771027, -0.77079951, 0.28547459, -0.6480673, -1.58786233]
        # self.robot.reset_joints()
        # self.robot.set_base_pos_orient(base_pos, base_orient)
        # self.robot.set_joint_angles(self.robot.controllable_joint_indices, joint_angles)
        # print(self.robot.get_pos_orient(self.robot.left_end_effector, False, True)[0])

        garments = ["hospital_gown", "tshirt_26", "tshirt_68", "tshirt_4", "tshirt_392"]
        garment_idx = self.garment_id #1
        garment = garments[garment_idx]
        self.garment = garment
        path_to_garment = os.path.join(self.directory, 'data/cloth3d/train/Tshirt', '{}.obj'.format(garment))
        output_path = os.path.join(self.directory, 'clothing', 'hospitalgown_reduced.obj')
        
        # mesh_in = o3d.io.read_triangle_mesh(path_to_garment)
        # mesh_in.compute_vertex_normals()

        # print(
        #     f'Input mesh has {len(mesh_in.vertices)} vertices and {len(mesh_in.triangles)} triangles'
        # )
        # o3d.visualization.draw_geometries([mesh_in])

        # voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 128
        # print(f'voxel_size = {voxel_size:e}')
        # mesh_smp = mesh_in.simplify_vertex_clustering(
        #     voxel_size=voxel_size,
        #     contraction=o3d.geometry.SimplificationContraction.Average)
        # print(
        #     f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
        # )

        # o3d.visualization.draw_geometries([mesh_smp])
        # o3d.io.write_triangle_mesh(output_path, mesh_smp)


        # self.cloth_attachment = self.create_sphere(radius=0.02, mass=0, pos=[0.4, -0.35, 1.05], visual=True, collision=False, rgba=[0, 0, 0, 1], maximal_coordinates=False)
        # self.cloth = p.loadSoftBody(path_to_garment, scale=cloth_scales[garment], mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=5, springDampingStiffness=0.01, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.0001, frictionCoeff=0.1, useFaceContact=1, physicsClientId=self.id)
        
        # self.cloth = p.loadSoftBody(path_to_garment, scale=cloth_scales[garment], mass=0.16, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=10, springDampingStiffness=0.1, springDampingAllDirections=0, springBendingStiffness=0.1, useSelfCollision=1, collisionMargin=0.001, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)
        # p.clothParams(self.cloth, kLST=0.055, kAST=1.0, kVST=0.5, kDP=0.01, kDG=10, kDF=0.39, kCHR=1.0, kKHR=1.0, kAHR=1.0, piterations=5, physicsClientId=self.id)
        self.cloth = p.loadSoftBody(path_to_garment, scale=cloth_scales[garment]*0.8, mass=0.16, useNeoHookean=self.useNeoHookean, useBendingSprings=1, useMassSpring=1, springElasticStiffness=self.elastic_stiffness, springDampingStiffness=self.damping_stiffness, springDampingAllDirections=self.all_direction, springBendingStiffness=self.bending_stiffnes, useSelfCollision=1, collisionMargin=0.001, frictionCoeff=0.1, useFaceContact=1, physicsClientId=self.id)
        p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 1], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.cloth, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=8, physicsClientId=self.id)

        vertex_index = picker_picking_particle_indices[garment][0]

        # NOTE: debug
        self.anchor_vertices = grasping_particle_indices[garment][::3]
        self.shoulder_polygon_indices = shoulder_polygon_particle_indices[garment]
        self.reward_triangle_idxes = polygon_triangle_indices[garment]

        # self.triangle1_point_indices = polygon_triangle_indices[garment][0]
        # self.triangle2_point_indices = polygon_triangle_indices[garment][1]
        # self.triangle3_point_indices = polygon_triangle_indices[garment][2]
        # self.triangle4_point_indices = polygon_triangle_indices[garment][3]

        # Move cloth grasping vertex into robot end effector
        if self.garment_id != 0:
            p.resetBasePositionAndOrientation(self.cloth, [0, 0, 0], self.get_quaternion([0, 0, np.pi/2]), physicsClientId=self.id)
        else:
            p.resetBasePositionAndOrientation(self.cloth, [0, 0, 0], self.get_quaternion([0, 0, np.pi]), physicsClientId=self.id)

        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        # data2 = p.getMeshData(self.cloth2, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        vertex_position = np.array(data[1][vertex_index])
        # vertex_position = np.array(data2[1][202])

        offset = self.robot.get_pos_orient(self.robot.left_end_effector)[0] - vertex_position
        
        if self.garment_id != 0:
            p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion([0, 0, np.pi/2]), physicsClientId=self.id)
        else:
            p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion([0, 0, np.pi]), physicsClientId=self.id)

        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        # data2 = p.getMeshData(self.cloth2, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)

        # new_vertex_position = np.array(data[1][vertex_index])

        # print('num vertices ', len(anchor_vertices))
        # print('len1', len(data[1]), 'len2', len(data2[1]))

        # new_vertices2 = []
        # smallest = np.inf  # Initialize smallest distance to infinity
        # all_vertices = anchor_vertices.append(vertex_index)
        # for v_idx in anchor_vertices:
        #     closest_idx = -1
        #     closest_dist = np.inf

        #     # Convert data[1][v_idx] to a NumPy array once to avoid redundant conversions
        #     anchor_point = np.array(data[1][v_idx])
            
        #     # Vectorized distance calculation
        #     distances = np.linalg.norm(np.array(data2[1]) - anchor_point, axis=1)

        #     # Find the closest point
        #     closest_idx = np.argmin(distances)  # Index of the closest point
        #     closest_dist = distances[closest_idx]  # Corresponding distance
            
        #     # Update smallest distance if necessary
        #     smallest = min(smallest, closest_dist)

        #     # Debug print: Check closest point coordinates and distance
        #     print(f"Closest point in data2: {data2[1][closest_idx]}, Anchor point in data: {data[1][v_idx]}")
        #     print(f"Distance: {closest_dist}")
            
        #     # Append the closest index to the new_vertices2 list
        #     new_vertices2.append(closest_idx)

        # # # Final debug prints
        # print(f"Total closest vertices: {len(new_vertices2)}, Smallest distance: {smallest}")
        # print(f"New vertices indices: {new_vertices2}")

        # NOTE: Create anchors between cloth and robot end effector
        p.createSoftBodyAnchor(self.cloth, vertex_index, self.robot.body, self.robot.left_end_effector, [0, 0, 0], physicsClientId=self.id)

        for i in self.anchor_vertices:
            pos_diff = np.array(data[1][i]) - vertex_position
            p.createSoftBodyAnchor(self.cloth, i, self.robot.body, self.robot.left_end_effector, pos_diff, physicsClientId=self.id)

        self.robot.enable_force_torque_sensor(self.robot.left_end_effector-1)

        # Disable collisions between robot and cloth
        for i in [-1] + self.robot.all_joint_indices:
            p.setCollisionFilterPair(self.robot.body, self.cloth, i, -1, 0, physicsClientId=self.id)
        p.setCollisionFilterPair(self.furniture.body, self.cloth, -1, -1, 0, physicsClientId=self.id)
        # Disable collision between chair and human
        for i in [-1] + self.human.all_joint_indices:
            p.setCollisionFilterPair(self.human.body, self.furniture.body, i, -1, 0, physicsClientId=self.id)

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        # p.setGravity(0, 0, 0, physicsClientId=self.id)
        
        # p.setGravity(0, 0, -9.81, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        # robot_euler = [-hand_ori[1], np.pi, hand_ori[0]]
        robot_euler = [0, np.pi, hand_ori[0]]

        joint_angles = self.robot.ik(self.robot.right_end_effector, hand_pos, robot_euler, self.robot.right_arm_ik_indices, max_iterations=1000, use_current_as_rest=True)
        self.robot.control(self.robot.controllable_joint_indices, joint_angles, self.robot.motor_gains, self.robot.motor_forces)
        p.stepSimulation(physicsClientId=self.id)

        # self.create_arm_motion()
        
        # Wait for the cloth to settle
        for i in range(20):
            print(i)
            p.stepSimulation(physicsClientId=self.id)
        print('Settled')

        robo_pos, robo_ori = self.robot.get_pos_orient(self.robot.left_end_effector)
        robo_pos[1] += 0.5

        joint_angles = self.robot.ik(self.robot.left_end_effector, robo_pos, robo_ori, self.robot.right_arm_ik_indices, max_iterations=1000, use_current_as_rest=False)
        self.robot.control(self.robot.controllable_joint_indices, joint_angles, self.robot.motor_gains, self.robot.motor_forces)
        for i in range(60):
            print(i)
            p.stepSimulation(physicsClientId=self.id)

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)   

        self.generate_arm_traj()
        self.time = time.time()
        self.init_env_variables()

        return self._get_obs()
    
    def generate_arm_traj(self):
        ik_indices = self.human.controllable_joint_indices
        joint_angles = self.human.get_joint_angles(ik_indices)
        if self.motion_id == 0:
            print('static arm') 
        
        elif self.motion_id == 1:
            with open('arm_motions/straight_up.pkl', 'rb') as f:
                target_joint_angles = pickle.load(f)
            steps = 60
            self.arm_traj = np.linspace(joint_angles, target_joint_angles, num=steps)
            self.repeat_traj = 0
            
        elif self.motion_id == 2:
            with open('arm_motions/bend_down.pkl', 'rb') as f:
                target_joint_angles = pickle.load(f)
            steps = 60
            self.arm_traj = np.linspace(joint_angles, target_joint_angles, num=steps)
            self.repeat_traj = 0
            
        elif self.motion_id == 3:
            pass
        elif self.motion_id == 4:
            with open('arm_motions/straight_down.pkl', 'rb') as f:
                target_joint_angles = pickle.load(f)
            steps = 60
            self.arm_traj = np.linspace(joint_angles, target_joint_angles, num=steps)
            self.repeat_traj = 0
        elif self.motion_id == 5:
            with open('arm_motions/reach_phone.pkl', 'rb') as f:
                target_joint_angles = pickle.load(f)
            steps = 60
            self.arm_traj = np.linspace(joint_angles, target_joint_angles, num=steps)
            self.repeat_traj = 1
        elif self.motion_id == 6:
            with open('arm_motions/receive_obj.pkl', 'rb') as f:
                target_joint_angles = pickle.load(f)
            steps = 60
            self.arm_traj = np.linspace(joint_angles, target_joint_angles, num=steps)
            self.repeat_traj = 1
        elif self.motion_id == 7:
            with open('arm_motions/scratch_head.pkl', 'rb') as f:
                target_joint_angles = pickle.load(f)
            steps = 60 #40 -> 30
            self.arm_traj = np.linspace(joint_angles, target_joint_angles, num=steps)
            self.repeat_traj = 1
        elif self.motion_id == 8:
            with open('arm_motions/maneki_arm.pkl', 'rb') as f:
                target_joint_angles = pickle.load(f)
            steps = 60
            self.arm_traj = np.linspace(joint_angles, target_joint_angles, num=steps)
            self.repeat_traj = 0

        else:
            print('Invalid motion id!')

    def generate_init_poses(self):
        ik_indices = self.human.controllable_joint_indices
        joint_angles = self.human.get_joint_angles(ik_indices)
        steps = 60

        with open('arm_motions/bend_down.pkl', 'rb') as f:
            motion2_target_joint_angles = pickle.load(f)
        self.poses_lst.extend(np.linspace(joint_angles, motion2_target_joint_angles, num=steps)[9::10])
        
        with open('arm_motions/straight_down.pkl', 'rb') as f:
            motion4_target_joint_angles = pickle.load(f)
        self.poses_lst.extend(np.linspace(joint_angles, motion4_target_joint_angles, num=steps)[9::10])

        with open('arm_motions/reach_phone.pkl', 'rb') as f:
            motion5_target_joint_angles = pickle.load(f)
        self.poses_lst.extend(np.linspace(joint_angles, motion5_target_joint_angles, num=steps)[9::10])

        with open('arm_motions/receive_obj.pkl', 'rb') as f:
            motion6_target_joint_angles = pickle.load(f)
        self.poses_lst.extend(np.linspace(joint_angles, motion6_target_joint_angles, num=steps)[9::10])

        with open('arm_motions/scratch_head.pkl', 'rb') as f:
            motion7_target_joint_angles = pickle.load(f)
        self.poses_lst.extend(np.linspace(joint_angles, motion7_target_joint_angles, num=steps)[9::10][:-2])

        with open('arm_motions/maneki_arm.pkl', 'rb') as f:
            motion8_target_joint_angles = pickle.load(f)
        self.poses_lst.extend(np.linspace(joint_angles, motion8_target_joint_angles, num=steps)[9::10])

        print('Num poses', len(self.poses_lst))

    def align_gripper_rot(self):
        rotation_scale = 0.002
        step = 0
        max_steps = 1000
        sign_z = 1
        sign_x = -1
        last_diff_x = None
        last_diff_y = None
        line_idx = alignment_line_indices[self.garment]
        stage = 0

        while stage != 2 and step < max_steps:
            if stage == 0:
                vec1 = self.line1_dir
                vec2 = np.array([0, 0, 1])
                normal = np.cross(vec1, vec2)
                x, y, z, cx, cy, cz, fx, fy, fz = p.getSoftBodyData(self.cloth, physicsClientId=self.id)
                mesh_points = np.concatenate([np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), np.expand_dims(z, axis=-1)], axis=-1)

                shoulder_point1, shoulder_point2 = mesh_points[line_idx[0]], mesh_points[line_idx[1]]
                shoulder_line = -shoulder_point2 + shoulder_point1
                shoulder_line_dir = shoulder_line / np.linalg.norm(shoulder_line)

                angle_between_normal_and_shoulder_line = np.arccos(np.dot(shoulder_line_dir, normal))
                angle_between_normal_and_shoulder_line = np.rad2deg(angle_between_normal_and_shoulder_line)

                diff_z = np.abs(angle_between_normal_and_shoulder_line - 90)
                finished = diff_z < 0.05

                if not finished:
                    if angle_between_normal_and_shoulder_line < 90:
                        action_z = sign_z * rotation_scale
                    else:
                        action_z = -sign_z * rotation_scale

                if last_diff_y is not None and diff_z > last_diff_y:
                    action_z = - action_z
                    sign_z = -sign_z
                
                last_diff_z = diff_z
                print('diff z', last_diff_z)
                if finished:
                    stage += 1
            
            if stage == 1:
                x, y, z, cx, cy, cz, fx, fy, fz = p.getSoftBodyData(self.cloth, physicsClientId=self.id)
                mesh_points = np.concatenate([np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), np.expand_dims(z, axis=-1)], axis=-1)

                shoulder_point1, shoulder_point2 = mesh_points[line_idx[0]], mesh_points[line_idx[1]]
                shoulder_line = -shoulder_point2 + shoulder_point1
                shoulder_line_dir = shoulder_line / np.linalg.norm(shoulder_line)

                diff = np.abs(np.arccos(np.dot(-self.line1_dir, shoulder_line_dir)))
                x_finished = diff < 0.05

                if not x_finished:
                    if diff > 0:
                        action_x = sign_x * rotation_scale
                    else:
                        action_x = -sign_x * rotation_scale                

                if last_diff_x is not None and diff > last_diff_x:
                    action_x = - action_x
                    sign_x = -sign_x
                
                last_diff_x = diff
                print('diff x', last_diff_x)

                if x_finished:
                    stage += 1

            pos, orient = self.robot.get_pos_orient(self.robot.right_end_effector)
            ee_cur_R = R.from_quat([orient[0], orient[1], orient[2], orient[3]])
            
            if last_diff_x is None:
                rotation_R = R.from_rotvec([0, 0, last_diff_z])
            else:
                rotation_R = R.from_rotvec([last_diff_x, 0, last_diff_z])
            new_ee_R = rotation_R * ee_cur_R
            new_ee_quat = new_ee_R.as_quat()
            for _ in range(1):
                agent_joint_angles = self.robot.ik(self.robot.right_end_effector, pos, new_ee_quat, self.robot.right_arm_ik_indices, max_iterations=200, use_current_as_rest=True)
                # self.robot.set_joint_angles(self.robot.controllable_joint_indices, agent_joint_angles)
                self.robot.control(self.robot.right_arm_ik_indices, agent_joint_angles, self.robot.motor_gains, self.robot.motor_forces)
                p.stepSimulation(physicsClientId=self.id)

            step += 1

    
    def create_arm_motion(self):
        ik_indices = self.human.controllable_joint_indices
        joint_angles = self.human.get_joint_angles(ik_indices)
        curr_pos, curr_orient = self.human.get_pos_orient(18)

        if self.motion_id == 1: # straight arm up down (need to adjust init robot pos)
            target_pos = curr_pos + np.array([-0.3, -0.1, -0.2])
            for i in range(10):
                print(i)
                target_joint_angles = self.human.ik(18, target_pos, target_orient=None, ik_indices=ik_indices, max_iterations=1000, use_current_as_rest=True)

                for _ in range(10):
                    self.human.control(ik_indices, target_joint_angles, 0.05, 100)
                    p.stepSimulation(physicsClientId=self.id)
                    print(self.human.get_pos_orient(18)[0])
                    joint_angles = self.human.get_joint_angles(ik_indices)
                    print('diff', np.linalg.norm(joint_angles-target_joint_angles))

                joint_angles = self.human.get_joint_angles(ik_indices)
                print('diff', np.linalg.norm(joint_angles-target_joint_angles))

                with open('arm_motions/arm_motion_{}.pkl'.format(self.motion_id), 'wb') as f:
                    pickle.dump(joint_angles, f)

        elif self.motion_id == 2:
            target_pos = curr_pos + np.array([0, 0, -0.3])
            for _ in range(8):
                target_joint_angles = self.human.ik(18, target_pos, target_orient=None, ik_indices=ik_indices, max_iterations=1000, use_current_as_rest=True)

                for _ in range(15):
                    self.human.control(ik_indices, target_joint_angles, 0.05, 100)
                    p.stepSimulation(physicsClientId=self.id)
                    print(self.human.get_pos_orient(18)[0])
                    joint_angles = self.human.get_joint_angles(ik_indices)
                    print('diff', np.linalg.norm(joint_angles-target_joint_angles))
                
                joint_angles = self.human.get_joint_angles(ik_indices)
                print('diff', np.linalg.norm(joint_angles-target_joint_angles))

            with open('arm_motions/arm_motion_{}.pkl'.format(self.motion_id), 'wb') as f:
                pickle.dump(joint_angles, f)
           
        elif self.motion_id == 3: # bend arm (need to adjust init robot pos)
            pass
        
        elif self.motion_id == 4:
            target_pos = curr_pos + np.array([0.1, -0.2, -0.3])
            for i in range(8):
                print(i)
                target_joint_angles = self.human.ik(18, target_pos, target_orient=None, ik_indices=ik_indices, max_iterations=1000, use_current_as_rest=True)

                for _ in range(15):
                    self.human.control(ik_indices, target_joint_angles, 0.05, 100)
                    p.stepSimulation(physicsClientId=self.id)
                    print(self.human.get_pos_orient(18)[0])
                    joint_angles = self.human.get_joint_angles(ik_indices)
                    print('diff', np.linalg.norm(joint_angles-target_joint_angles))

                
                joint_angles = self.human.get_joint_angles(ik_indices)
                print('diff', np.linalg.norm(joint_angles-target_joint_angles))

            with open('arm_motions/arm_motion_{}.pkl'.format(self.motion_id), 'wb') as f:
                pickle.dump(joint_angles, f)

        elif self.motion_id == 5: # reach for phone in pocket (bent->down->bent)
            target_pos = curr_pos + np.array([0.4, 0.1, -0.5])
            for i in range(3):
                print(i)
                target_joint_angles = self.human.ik(18, target_pos, target_orient=None, ik_indices=ik_indices, max_iterations=1000, use_current_as_rest=True)

                for _ in range(10):
                    self.human.control(ik_indices, target_joint_angles, 0.05, 100)
                    p.stepSimulation(physicsClientId=self.id)
                    print(self.human.get_pos_orient(18)[0])
                    joint_angles = self.human.get_joint_angles(ik_indices)
                    print('diff', np.linalg.norm(joint_angles-target_joint_angles))
                
                joint_angles = self.human.get_joint_angles(ik_indices)
                print('diff', np.linalg.norm(joint_angles-target_joint_angles))
                with open('arm_motions/arm_motion_{}.pkl'.format(self.motion_id), 'wb') as f:
                    pickle.dump(joint_angles, f)

        elif self.motion_id == 6: # receive an item (bent->straight->bent)
            target_pos = curr_pos + np.array([-0.2, -0.1, -0.2])
            for i in range(10):
                print(i)
                target_joint_angles = self.human.ik(18, target_pos, target_orient=None, ik_indices=ik_indices, max_iterations=1000, use_current_as_rest=True)

                for _ in range(10):
                    self.human.control(ik_indices, target_joint_angles, 0.05, 100)
                    p.stepSimulation(physicsClientId=self.id)
                    print(self.human.get_pos_orient(18)[0])
                    joint_angles = self.human.get_joint_angles(ik_indices)
                    print('diff', np.linalg.norm(joint_angles-target_joint_angles))

                
                joint_angles = self.human.get_joint_angles(ik_indices)
                print('diff', np.linalg.norm(joint_angles-target_joint_angles))

                with open('arm_motions/arm_motion_{}.pkl'.format(self.motion_id), 'wb') as f:
                    pickle.dump(joint_angles, f)

        elif self.motion_id == 7: # scratch face (bent->up->bent)
            target_pos = curr_pos + np.array([0.5, 0.3, 0.3])
            for i in range(10):
                print(i)
                target_joint_angles = self.human.ik(18, target_pos, target_orient=None, ik_indices=ik_indices, max_iterations=1000, use_current_as_rest=True)

                for _ in range(10):
                    self.human.control(ik_indices, target_joint_angles, 0.05, 100)
                    p.stepSimulation(physicsClientId=self.id)
                    print(self.human.get_pos_orient(18)[0])
                    joint_angles = self.human.get_joint_angles(ik_indices)
                    print('diff', np.linalg.norm(joint_angles-target_joint_angles))
                
                joint_angles = self.human.get_joint_angles(ik_indices)
                print('diff', np.linalg.norm(joint_angles-target_joint_angles))
                with open('arm_motions/arm_motion_{}.pkl'.format(self.motion_id), 'wb') as f:
                    pickle.dump(joint_angles, f)

        elif self.motion_id == 8:
            target_pos = curr_pos + np.array([0.1, -0.15, 0.25])
            for i in range(10):
                print(i)
                target_joint_angles = self.human.ik(18, target_pos, target_orient=None, ik_indices=ik_indices, max_iterations=1000, use_current_as_rest=True)

                for _ in range(10):
                    self.human.control(ik_indices, target_joint_angles, 0.05, 100)
                    p.stepSimulation(physicsClientId=self.id)
                    print(self.human.get_pos_orient(18)[0])
                    joint_angles = self.human.get_joint_angles(ik_indices)
                    print('diff', np.linalg.norm(joint_angles-target_joint_angles))

                
                joint_angles = self.human.get_joint_angles(ik_indices)
                print('diff', np.linalg.norm(joint_angles-target_joint_angles))

                with open('arm_motions/arm_motion_{}.pkl'.format(self.motion_id), 'wb') as f:
                    pickle.dump(joint_angles, f)

        else:
            print('Invalid motion id')



            


