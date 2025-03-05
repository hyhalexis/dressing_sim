from dressing import DressingEnv
# from dressing_data import DressingEnvData

from agents import sawyer, human
from agents.sawyer import Sawyer
from agents.human import Human
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm = 'right'
human_controllable_joint_indices = human.right_arm_joints

class DressingSawyerEnv(DressingEnv):
    def __init__(self):
        super(DressingSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DressingSawyerHumanEnv(DressingEnv, MultiAgentEnv):
    def __init__(self, **kwargs):
        super(DressingSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True), **kwargs)
register_env('assistive_gym:DressingSawyerHuman-v1', lambda config: DressingSawyerHumanEnv())

# class DressingSawyerHumanEnvData(DressingEnvData, MultiAgentEnv):
#     def __init__(self, **kwargs):
#         super(DressingSawyerHumanEnvData, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True), **kwargs)
# register_env('assistive_gym:DressingSawyerHumanEnvDataCollection', lambda config: DressingSawyerHumanEnvData())
