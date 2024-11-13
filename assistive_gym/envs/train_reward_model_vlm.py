import pickle as pkl
from pc_replay_buffer import PointCloudReplayBuffer
from chester.run_exp import run_experiment_lite, VariantGenerator
import json
import os
import copy
from chester import logger
import utils
from default_config import DEFAULT_CONFIG
from logger import Logger
from SAC_AWAC import SAC_AWACAgent
from visualization import save_numpy_as_gif
import time
import datetime
import dateutil.tz
from collections import defaultdict
import copy
import json
import multiprocessing as mp
import time
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import wandb
from torch_geometric.data import Batch
from reward_model import RewardModelImage
from collections import deque
from dressing_envs import DressingSawyerHumanEnv

def create_vg_from_json(json_file):
    assert json_file is not None
    with open(json_file, 'r') as f:
        variant_dict = json.load(f)

    vg = VariantGenerator()
    for key, value in variant_dict.items():
        vg.add(key, [value])

    return vg

def vv_to_args(vv):
    args = VArgs(vv)

    # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    return args


class VArgs(object):
    def __init__(self, vv):
        for key, val in vv.items():
            setattr(self, key, val)

def run_task(vv, log_dir=None, exp_name=None):
    # if resume from directory
    # log_dir = "/home/sreyas/Desktop/Projects/softagent_rpad/data/reward_collection/"
    if vv['resume_from_ckpt']:
        log_dir = vv['resume_from_exp_path']
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir, exp_name=exp_name, format_strs=['csv'])
    logdir = logger.get_dir()
    print("Logging at ", log_dir)
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)
    updated_vv = copy.copy(DEFAULT_CONFIG)
    vv = vv.variants()[-1]
    updated_vv.update(**vv)
    if updated_vv['use_wandb']:
        wandb.init(project="dressing-flow", name=exp_name, entity="conan-iitkgp", config=updated_vv, settings=wandb.Settings(start_method='fork'))


    # pr = cProfile.Profile()
    # pr.enable()
    # main(vv_to_args(updated_vv))
    train_reward_model(vv_to_args(updated_vv))
    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    # ps.print_stats(50)
    # print(s.getvalue())
    # ps = pstats.Stats(pr, stream=s).sort_stats('time')
    # ps.print_stats(50)
    # print(s.getvalue())


def update_env_kwargs(vv):
    new_vv = vv.copy()
    for v in vv:
        if v.startswith('env_kwargs_'):
            arg_name = v[len('env_kwargs_'):]
            new_vv['env_kwargs'][arg_name] = vv[v]
            del new_vv[v]
    return new_vv


def learn_reward(model, reward_update = 200):
    model.train()
    model
    labeled_queries = model.uniform_sampling()
    
    if labeled_queries > 0:
        for epoch in range(reward_update):
            train_acc = model.train_reward()
            total_acc = np.mean(train_acc)

            if total_acc > 0.97:
                break
        
        print("Reward function is updated!! ACC: " + str(total_acc))
    
    return total_acc

def train_reward_model(args): 
    utils.set_seed_everywhere(args.seed)
    num_train_steps = 1000
    save_interval = 10

    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs
    reward_schedule = 3#Reward model change_batch method parameter
    # build environment
    env = DressingSawyerHumanEnv(render=args.render)

    gripper_idx = 2
    # make directory for logging
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    args.pc_feature_dim = 3
    work_dir = "/home/sreyas/Desktop/Projects/softagent_rpad/data/local/2024-0707-data_collection-unoccluded-13_debug/2024-0707-data_collection-unoccluded-13_debug-07_08_13_44_16-001" #logger.get_dir()
    work_dir = utils.make_dir(os.path.join(work_dir, 'buffer'))
    args.work_dir =logger.get_dir()
 
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))
    L = Logger(args.work_dir, use_tb=args.save_tb, chester_logger=logger)

    # create replay buffer
    device = torch.device('cuda:{}'.format(args.cuda_idx) if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    action_shape = [6]
    obs_shape = env.observation_space.shape
    rb_limit = args.replay_buffer_capacity // args.replay_buffer_num
    data = {}
    data["observations"] = []
    data["images"] = []
    data["actions"] = []
    data["next_observations"] = []
    data["next images"] = []
    data["terminals"] = []
    data_dir = "/home/sreyas/Desktop/Projects/dressing_real_world-master/src/dressing/code/offline_rl/offline_data_pickled"
    file_nos = os.listdir(data_dir)
    file_nos.remove("videos")
    file_nos = sorted(file_nos)
    file_nos = [os.path.join(data_dir, file_no) for file_no in file_nos]
    
    for i in tqdm(range(len(file_nos))):
        file = file_nos[i]
        with open(file, 'rb') as f:
            data_dict = pkl.load(f)
        
        data["observations"] += data_dict["observations"]
        data["images"] += list(data_dict["images"])
        data["actions"] += list(data_dict["actions"])
        # data["next_observations"] += data_dict["next_observations"]
        # data["next images"] += data_dict["next images"].tolist()
        # data["terminals"] += data_dict["terminals"].tolist()
    
  
    reward_model = RewardModelImage(obs_shape, action_shape, args, use_action=False)

    data_size = len(data["observations"])
    ##### We will load the transitions from the point cloud buffer to the reward model buffer
   
    for i in tqdm(range(data_size)):
        # if not buffer.not_dones[i] : print(" done at ", i)
        obs = data["observations"][i]
        action = data["actions"][i]
        reward_model.add_data(obs, action, img=data["images"][i])
        
    model_save_dir = os.path.join(args.work_dir, "models")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print("Buffer size: ", len(reward_model.inputs))
    print("Buffer loaded to reward model")
    # store train returns of recent 10 episodes
    avg_train_true_return = deque([], maxlen=10) 
    start_time = time.time()
    reward_learning_acc = 0
    step_ = 0
    segment = 1
    # reward_model.prepare_test_data()
    for i in tqdm(range(15)):
        reward_model.uniform_sampling()
    # reward_model.uniform_sampling()
    print("inital buffer data added to reward model")
    for step in tqdm(range(step_,  num_train_steps)):
        if reward_schedule == 1:
            frac = (num_train_steps-step) / num_train_steps
            if frac == 0:
                frac = 0.01
        elif reward_schedule == 2:
            frac = num_train_steps / (num_train_steps-step +1)
        else:
            frac = 1
        reward_model.change_batch(frac)
        
        # update margin --> not necessary / will be updated soon
        new_margin = np.mean(avg_train_true_return) * (segment / 250)
        reward_model.set_teacher_thres_skip(new_margin)
        reward_model.set_teacher_thres_equal(new_margin)
                
 
        train_acc = reward_model.train_reward()
        reward_learning_acc = np.mean(train_acc)
        L.log('train/acc', reward_learning_acc, step)
        if args.use_wandb: wandb.log({'train/acc': reward_learning_acc, 'step': step})

        step += 1
        
        if step % save_interval == 0 and step > 0:
            print("Saving model at step ", step)
            print("Reward learning acc: ", reward_learning_acc)
            reward_model.save(model_save_dir, step)
            # reward_model.eval_test_data()
            # labeled_queries = reward_model.uniform_sampling()
            # if labeled_queries <= 0: 
            #     raise ValueError("No labeled queries")
        
    reward_model.save(model_save_dir, step)

    
exp_prefix = "debug" #'2024-0728-real-model-experiments'
load_variant_path = "/home/sreyas/Desktop/Projects/softagent_rpad/dummy/variant_reward.json"
loaded_vg = create_vg_from_json(load_variant_path)
print("Loaded configs from ", load_variant_path)
vg = loaded_vg
print(vg)
print('Number of configurations: ', len(vg.variants()))
now = datetime.datetime.now(dateutil.tz.tzlocal())

exp_count = 0
timestamp = now.strftime('%m_%d_%H_%M_%S')
exp_name = "real_world_vlm_queried"
print(exp_name)
exp_name = "{}-{}-{:03}".format(exp_name, timestamp, exp_count)
log_dir = "data" + "/local/" + exp_prefix + "/" + exp_name

run_task(vg, log_dir=log_dir, exp_name=exp_name)