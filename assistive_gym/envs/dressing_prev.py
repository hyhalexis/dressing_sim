import os, time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from env import AssistiveEnv
from .agents.human_mesh import HumanMesh
from garment_idx_utils import *
from PIL import Image
import open3d as o3d

class DressingEnv(AssistiveEnv):
    def __init__(self, robot, human, use_ik=False):
        super(DressingEnv, self).__init__(robot=robot, human=human, task='dressing', obs_robot_len=(16 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(16 + (len(human.controllable_joint_indices) if human is not None else 0)), frame_skip=1, time_step=0.02, deformable=True)
        self.use_ik = use_ik
        self.use_mesh = (human is None)
        self.arm_traj_idx = 0
        self.arm_traj = None
        self.traj_direction = 1

        # 1 skip, 0.1 step, 50 substep, springElasticStiffness=100 # ? FPS
        # 1 skip, 0.1 step, 20 substep, springElasticStiffness=10 # 4.75 FPS
        # 5 skip, 0.02 step, 4 substep, springElasticStiffness=5 # 5.75 FPS
        # 10 skip, 0.01 step, 2 substep, springElasticStiffness=5 # 4.5 FPS

    def step(self, action):
        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        # action = np.ones(7)
        if self.use_ik:
            self.take_step(action, action_multiplier=0.01, ik=True)
        else:
            self.take_step(action, action_multiplier=0.003)

        shoulder_pos = self.human.get_pos_orient(self.human.right_upperarm)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_forearm)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_hand)[0]

        # Get cloth data
        x, y, z, cx, cy, cz, fx, fy, fz = p.getSoftBodyData(self.cloth, physicsClientId=self.id)
        mesh_points = np.concatenate([np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), np.expand_dims(z, axis=-1)], axis=-1)
        # forces = np.concatenate([np.expand_dims(fx, axis=-1), np.expand_dims(fy, axis=-1), np.expand_dims(fz, axis=-1)], axis=-1) * 10
        # contact_positions = np.concatenate([np.expand_dims(cx, axis=-1), np.expand_dims(cy, axis=-1), np.expand_dims(cz, axis=-1)], axis=-1)
        # Get 3D points for two triangles around the sleeve to detect if the sleeve is around the arm
        triangle1_points = mesh_points[self.triangle1_point_indices]
        triangle2_points = mesh_points[self.triangle2_point_indices]
        triangle3_points = mesh_points[self.triangle3_point_indices]
        triangle4_points = mesh_points[self.triangle4_point_indices]

        # TODO: fix to make it work with four triangles
        forearm_in_sleeve, upperarm_in_sleeve, distance_along_forearm, distance_along_upperarm, distance_to_hand, distance_to_elbow, distance_to_shoulder, forearm_length, upperarm_length = self.util.sleeve_on_arm_reward(triangle1_points, triangle2_points, shoulder_pos, elbow_pos, wrist_pos, self.human.hand_radius, self.human.elbow_radius, self.human.shoulder_radius)
        self.forearm_in_sleeve = forearm_in_sleeve
        self.upperarm_in_sleeve = upperarm_in_sleeve

        reward_action = -np.linalg.norm(action) # Penalize actions
        if self.upperarm_in_sleeve:
            reward_dressing = forearm_length
            if distance_along_upperarm < upperarm_length:
                reward_dressing += distance_along_upperarm
        elif self.forearm_in_sleeve and distance_along_forearm < forearm_length:
            reward_dressing = distance_along_forearm
        else:
            reward_dressing = -distance_to_hand

        reward = self.config('dressing_reward_weight')*reward_dressing + self.config('action_weight')*reward_action

        obs = self._get_obs()

        if reward_dressing > self.task_success:
            self.task_success = reward_dressing

        if self.gui:
            print('Task success:', self.task_success, 'Dressing reward:', reward_dressing)

        info = {'total_force_on_human': 0, 'task_success': min(self.task_success / self.config('task_success_threshold'), 1), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}

    def _get_obs(self, agent=None):
        end_effector_pos, end_effector_orient = self.robot.get_pos_orient(self.robot.left_end_effector)
        end_effector_pos_real, end_effector_orient_real = self.robot.convert_to_realworld(end_effector_pos, end_effector_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        shoulder_pos = self.human.get_pos_orient(self.human.right_upperarm)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_forearm)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_hand)[0]
        shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)
        # self.cloth_force_sum = np.sum(np.linalg.norm(self.cloth_forces, axis=-1))
        # self.robot_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        # self.total_force_on_human = self.robot_force_on_human + self.cloth_force_sum
        robot_obs = np.concatenate([end_effector_pos_real, end_effector_orient_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real]).ravel()

        self.setup_camera_rpy(camera_target=[-0.37448014, -0.80101663, 1.125], distance=0.01, rpy=[0, 0, 0], fov=60, camera_width=360, camera_height=360)

        # img, depth, seg_mask = self.get_camera_image_depth()

        # image = Image.fromarray(img[..., :3], 'RGB')
        # image.save('./image.png')
        # plt.imshow(depth, cmap='gray')
        # # print('saved')

        points, colors = self.get_point_cloud(self.human)

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
        if points is not None and len(points) > 0:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

            # o3d.visualization.draw_geometries([point_cloud, axes])
        
        # image = Image.fromarray(img[..., :3], 'RGB')
        # image.save('./captured_image.png')
        # plt.imshow(depth, cmap='gray')
        # print('saved')
        # # plt.imshow(image)
        # plt.show()

        # TODO: Make this return pc
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            end_effector_pos_human, end_effector_orient_human = self.human.convert_to_realworld(end_effector_pos, end_effector_orient)
            shoulder_pos_human, _ = self.human.convert_to_realworld(shoulder_pos)
            elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
            wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)
            human_obs = np.concatenate([end_effector_pos_human, end_effector_orient_human, human_joint_angles, shoulder_pos_human, elbow_pos_human, wrist_pos_human]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def reset(self):
        super(DressingEnv, self).reset()
        self.build_assistive_env('wheelchair_left', gender='female', human_impairment='tremor')
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, np.pi/2.0])

        # Update robot and human motor gains
        self.human.motor_gains = 0.1

        self.robot.motor_gains = 0.05
        self.robot.motor_forces = 100.0

        if self.use_mesh:
            self.human = HumanMesh()
            joints_positions = [(self.human.j_right_shoulder_z, 60), (self.human.j_right_elbow_y, 90), (self.human.j_left_shoulder_z, -10), (self.human.j_left_elbow_y, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee_x, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee_x, 80)]
            body_shape = np.zeros((1, 10))
            gender = 'female' # 'random'
            self.human.init(self.directory, self.id, self.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions, left_hand_pose=[[-2, 0, 0, -2, 0, 0]])

            chair_seat_position = np.array([0, 0.1, 0.55])
            self.human.set_base_pos_orient(chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])
        else:
            joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_right_shoulder_x, 80), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
            self.human.set_joint_angles([j for j, _ in joints_positions], [np.deg2rad(j_angle) for _, j_angle in joints_positions])
            self.human.target_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.05, 100)

        # if self.human.controllable or self.human.impairment == 'tremor':
        #     self.human.control(self.human.controllable_joint_indices, self.human.get_joint_angles(self.human.controllable_joint_indices), 0.05, 1)

        shoulder_pos = self.human.get_pos_orient(self.human.right_upperarm)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_forearm)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_hand)[0]

        # target_ee_pos = np.array([0.45, -0.45, 1.05])# + self.np_random.uniform(-0.05, 0.05, size=3)
        # target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task][0])
        # target_ee_orient_shoulder = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task][-1])
        # offset = np.array([0, 0, 0.1])
        # self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient)], [(shoulder_pos+offset, target_ee_orient_shoulder), (elbow_pos+offset, target_ee_orient), (wrist_pos+offset, target_ee_orient)], arm='left', tools=[], collision_objects=[self.human, self.furniture], right_side=False)
        # print(self.robot.get_base_pos_orient())
        # print(self.robot.get_joint_angles(self.robot.controllable_joint_indices))
        base_pos = [-1.17448014, -0.30101663, 0.825]
        base_orient = [0., 0., 0., 1.]

        # base_orient = [0., 0., -0.25249673, 0.96759775]
        joint_angles = [-1.64317203, -0.95115511, -1.96771027, -0.77079951, 0.28547459, -0.6480673, -1.58786233]
        self.robot.reset_joints()
        self.robot.set_base_pos_orient(base_pos, base_orient)
        self.robot.set_joint_angles(self.robot.controllable_joint_indices, joint_angles)

        garments = ["hospital_gown", "tshirt_26", "tshirt_68", "tshirt_4", "tshirt_392"]
        garment_idx = 2
        garment = garments[garment_idx]
        path_to_garment = os.path.join(self.directory, 'data/cloth3d/train/Tshirt', '{}.obj'.format(garment))
        output_path = os.path.join(self.directory, 'clothing', 'hospitalgown_reduced.obj')
        # mesh_in = o3d.io.read_triangle_mesh(path_to_garment)
        # mesh_in.compute_vertex_normals()

        # print(
        #     f'Input mesh has {len(mesh_in.vertices)} vertices and {len(mesh_in.triangles)} triangles'
        # )
        # # o3d.visualization.draw_geometries([mesh_in])

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
        self.cloth = p.loadSoftBody(path_to_garment, scale=1.0, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=5, springDampingStiffness=0.01, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.0001, frictionCoeff=0.1, useFaceContact=1, physicsClientId=self.id)
        # self.cloth2 = p.loadSoftBody(output_path, scale=1.0, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=5, springDampingStiffness=0.01, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.0001, frictionCoeff=0.1, useFaceContact=1, physicsClientId=self.id)

        print('loaded cloth')
        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'data', 'Tshirt-4.obj'), scale=cloth_scales[garment], mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=5, springDampingStiffness=0.01, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.0001, frictionCoeff=0.1, useFaceContact=1, physicsClientId=self.id)
        p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 0.5], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.cloth, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=8, physicsClientId=self.id)

        vertex_index = picker_picking_particle_indices[garment][0]

        anchor_vertices = grasping_particle_indices[garment]
        self.triangle1_point_indices = polygon_triangle_indices[garment][0]
        self.triangle2_point_indices = polygon_triangle_indices[garment][1]
        self.triangle3_point_indices = polygon_triangle_indices[garment][2]
        self.triangle4_point_indices = polygon_triangle_indices[garment][3]

        # Move cloth grasping vertex into robot end effector
        p.resetBasePositionAndOrientation(self.cloth, [0, 0, 0], self.get_quaternion([0, 0, np.pi]), physicsClientId=self.id)
        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        # data2 = p.getMeshData(self.cloth2, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        vertex_position = np.array(data[1][vertex_index])
        # vertex_position = np.array(data2[1][202])

        offset = self.robot.get_pos_orient(self.robot.left_end_effector)[0] - vertex_position
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

        for i in anchor_vertices:
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

        # if not self.robot.mobile:
        #     self.robot.set_gravity(0, 0, 0)
        # self.human.set_gravity(0, 0, 0)
        # p.setGravity(0, 0, 0, physicsClientId=self.id)
        
        # p.setGravity(0, 0, -9.81, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Wait for the cloth to settle
        for i in range(30):
            print(i)
            p.stepSimulation(physicsClientId=self.id)
        # print('Settled')
        ik_indices = self.human.controllable_joint_indices

        curr_pos, curr_orient = self.human.get_pos_orient(18)
        target_pos = curr_pos + np.array([0, 0, 0.1])
        target_orient = curr_orient
        steps = 10
        inter_pos = np.linspace(curr_pos, target_pos, num=steps)
        print('init', inter_pos)
        joints_traj = []
        for pos in inter_pos:
            joints_pos = self.human.ik(18, pos, target_orient, ik_indices, max_iterations=200, use_current_as_rest=True)
            joints_traj.append(joints_pos)
        self.arm_traj = np.array(joints_traj)

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)

        

        # target_joint_angles = self.human.ik(18, target_pos, target_orient, ik_indices, max_iterations=200, use_current_as_rest=True)
        

        # ik_indices = self.human.controllable_joint_indices
        # joint_angles = self.human.get_joint_angles(ik_indices)
        # curr_pos, curr_orient = self.human.get_pos_orient(18)

        # print('init', curr_pos)

        # target_pos = curr_pos + np.array([0, 0, 0.1])
        # target_orient = curr_orient
        # steps = 10

        # target_joint_angles = self.human.ik(18, target_pos, target_orient, ik_indices, max_iterations=200, use_current_as_rest=True)

        # self.arm_traj = np.linspace(joint_angles, target_joint_angles, num=steps)

        self.time = time.time()
        self.init_env_variables()
        return self._get_obs()

