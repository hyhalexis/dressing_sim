import pickle as pkl
from chester.run_exp import run_experiment_lite, VariantGenerator
import json
import os
import copy
from chester import logger
import utils
from default_config import DEFAULT_CONFIG
from logger import Logger
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
from reward_model import RewardModelVLM
from collections import deque

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
        wandb.init()
        # wandb.init(project="cb_impl/dressing", name=exp_name, entity="conan-iitkgp", config=updated_vv, settings=wandb.Settings(start_method='fork'))
    train_reward_model(vv_to_args(updated_vv))

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
        for _ in range(reward_update):
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

    # make directory for logging
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    args.pc_feature_dim = 4
    
    args.work_dir =logger.get_dir()
    print(args.work_dir)
 
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))
    L = Logger(args.work_dir, use_tb=args.save_tb, chester_logger=logger)

    # create replay buffer
    device = torch.device('cuda:{}'.format(args.cuda_idx) if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    action_shape = [6]
    obs_shape = (30000,)
  
    data_dir = '/scratch/alexis/data/traj_data_with_force_one-hot/labeled'  
    reward_model = RewardModelVLM(obs_shape, action_shape, args, use_action=False)
    reward_model.add_data_from_cache(data_dir)

    model_save_dir = os.path.join(args.work_dir, "models")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        
    # store train returns of recent 10 episodes
    avg_train_true_return = deque([], maxlen=10) 
    start_time = time.time()
    reward_learning_acc = 0
    step_ = 0
    segment = 1
   
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
            reward_model.save(model_save_dir, step, reward_learning_acc)
            # reward_model.eval_test_data()
            # labeled_queries = reward_model.uniform_sampling()
            # if labeled_queries <= 0: 
            #     raise ValueError("No labeled queries")
        
    reward_model.save(model_save_dir, step, reward_learning_acc)

    
exp_prefix = "12-25-reward-training" #'2024-0728-real-model-experiments'
load_variant_path = "/home/alexis/assistive-gym-film/assistive_gym/envs/variant_reward_vlm.json"
loaded_vg = create_vg_from_json(load_variant_path)
print("Loaded configs from ", load_variant_path)
vg = loaded_vg
print(vg)
print('Number of configurations: ', len(vg.variants()))
now = datetime.datetime.now(dateutil.tz.tzlocal())

exp_count = 0
timestamp = now.strftime('%m_%d_%H_%M_%S')
exp_name = "reward_model_sim_training_with_force_one-hot"
print(exp_name)
exp_name = "{}-{}-{:03}".format(exp_name, timestamp, exp_count)
log_dir = "/scratch/alexis/data/" + exp_prefix + "/" + exp_name

run_task(vg, log_dir=log_dir, exp_name=exp_name)