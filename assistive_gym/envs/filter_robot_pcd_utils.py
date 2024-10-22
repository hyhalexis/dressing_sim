import numpy as np
import pybullet as p
import os.path as osp
import time


class FilterRobotPCDInterface():
    def __init__(self, gui=False):
        super().__init__()
        self.gui = gui
        if self.gui:
            try:
                self.id = p.connect(p.GUI)
            except:
                self.id = p.connect(p.DIRECT)
        else:
            self.id = p.connect(p.DIRECT)
        self.gravity = -9.81
        p.setTimeStep(1/240, physicsClientId=self.id)
        p.resetSimulation(physicsClientId=self.id)
        if self.gui:
            p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.id)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.id)
        self.asset_dir = osp.join("/data/robogen/RoboGen-sim2real/manipulation", "assets/")
        
        self.load_robot()

        self.initial_robot_link_pcs = []
        for i in range(12):
            link_pc = np.load(f"/data/robogen/RoboGen-sim2real/real_world/filter_robot_pointcloud/robot_pointcloud/link_{i}.npy")
            self.initial_robot_link_pcs.append(link_pc)

    def load_robot(self):
      
        # Create robot
        # self.robot = Panda(slider=False)
        # self.robot.init(self.asset_dir, self.id, 0, fixed_base=True, use_suction=False)
        self.panda = p.loadURDF("/data/robogen/RoboGen-sim2real/manipulation/assets/panda_bullet/panda.urdf", 
                                useFixedBase=True, basePosition=[0, 0, 0], 
                                flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)


    def get_robot_pc(self):
        robot_link_pcs = []
        for link in range(12):
            res = p.getLinkState(self.panda, link, physicsClientId=self.id)
            pos = res[0]
            orient = res[1]

            T_body_to_world = np.eye(4)
            T_body_to_world[:3, :3] = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)
            T_body_to_world[:3, 3] = pos
            point_cloud = self.initial_robot_link_pcs[link].reshape(-1, 3)
            point_cloud_homogeneous = np.concatenate([point_cloud, np.ones((point_cloud.shape[0], 1))], axis=1)
            transformed_pc_homogeneous = (T_body_to_world @ point_cloud_homogeneous.T).T
            transformed_pc = transformed_pc_homogeneous[:, :3]
            robot_link_pcs.append(transformed_pc)

        robot_pc = np.concatenate(robot_link_pcs, axis=0)
        return robot_pc

    def set_joint_angles(self, joint_angles, gripper_angle):
        for idx in range(len(joint_angles)):
            p.resetJointState(self.panda, idx, joint_angles[idx], physicsClientId=self.id)

        p.resetJointState(self.panda, 9, gripper_angle, physicsClientId=self.id)
        p.resetJointState(self.panda, 10, gripper_angle, physicsClientId=self.id)



if __name__ == "__main__":
    interface = FilterRobotPCDInterface(gui=True)
    time1 = time.time()
    input_joint_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    interface.robot.set_joint_angles(interface.robot.right_arm_joint_indices, input_joint_values)
    robot_pc = interface.get_robot_pc(interface.robot.body, physicsClientId=interface.id)
    time2 = time.time()
    print("==========time======", time2 - time1)
    import pdb; pdb.set_trace()
    p.addUserDebugPoints(robot_pc, [[1, 0, 0] for _ in range(len(robot_pc))], physicsClientId=interface.id)


    

