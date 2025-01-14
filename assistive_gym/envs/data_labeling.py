import pickle as pkl
from pc_replay_buffer import PointCloudReplayBuffer
from chester.run_exp import run_experiment_lite, VariantGenerator
import json
import os
import copy
from chester import logger
import utils
from default_config import DEFAULT_CONFIG
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
    # os.makedirs(logdir, exist_ok=True)
    updated_vv = copy.copy(DEFAULT_CONFIG)
    vv = vv.variants()[-1]
    updated_vv.update(**vv)

    train_reward_model(vv_to_args(updated_vv))


def update_env_kwargs(vv):
    new_vv = vv.copy()
    for v in vv:
        if v.startswith('env_kwargs_'):
            arg_name = v[len('env_kwargs_'):]
            new_vv['env_kwargs'][arg_name] = vv[v]
            del new_vv[v]
    return new_vv

def gripper_shoulder_dist(gripper_pos, line_points, shoulder_offset=[0, 0, 0.11], y_scale=1):
    gripper_pos[1] *= y_scale
    target_pos = np.array(line_points[2]) + np.array(shoulder_offset)
    dist = np.linalg.norm(target_pos-gripper_pos)
    return dist


def min_distance(r: np.ndarray, a: np.ndarray):
    """ Compute the minimal distance between a point and a segment.

    Given a segment of points xa and xb and a point p

    Parameters
    ----------
    r
        xb - xa

    a
        xa - p

    Returns
    -------
    d
        The minimal distance spanning from p to the segment
    """

    min_t = np.clip(-a.dot(r) / (r.dot(r)), 0, 1)

    d = a + min_t * r

    return np.sqrt(d.dot(d))

def gripper_arm_dist(gripper_pos, line_points, offset = [0, 0, 0.11]):
    line_points += offset
    dist_forearm = min_distance(line_points[1]-line_points[0], line_points[0]-gripper_pos)
    dist_upperarm = min_distance(line_points[2]-line_points[1], line_points[1]-gripper_pos)
    # print(dist_forearm, dist_upperarm)
    return min(dist_forearm, dist_upperarm)


def get_label(vlm_dict_list_1, vlm_dict_list_2, batch_number):
    # get the label for the pair of obs_1, act_1 and obs_2, act_2
    
    # equally preferable
    combined_images_list = []

    # save_path = "traj_data_labeled"
    save_path = "temp"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    assert len(vlm_dict_list_1) == len(vlm_dict_list_2)
    gt_labels = []
    pref_labels1 = []
    pref_labels2 = []
    pref_labels3 = []


    # print('Dict len: ', len(vlm_dict_list_1))
    for idx in range(len(vlm_dict_list_1)):
        img1 = vlm_dict_list_1[idx]["img"]
        img2 = vlm_dict_list_2[idx]["img"]
        
        combined_image = np.concatenate([img1, img2], axis=1)
        combined_images_list.append(combined_image)
        combined_image = Image.fromarray(combined_image)
        first_image_save_path = os.path.join(save_path, "first_{:06}.png".format(idx))
        second_image_save_path = os.path.join(save_path, "second_{:06}.png".format(idx))
        Image.fromarray(img1).save(first_image_save_path)
        Image.fromarray(img2).save(second_image_save_path)            
            
        if vlm_dict_list_1[idx]["time_steps"] > vlm_dict_list_2[idx]["time_steps"]:
            gt_labels.append(0)
        elif vlm_dict_list_1[idx]["time_steps"] < vlm_dict_list_2[idx]["time_steps"]:
            gt_labels.append(1)
        else:
            gt_labels.append(-1)
                
        if vlm_dict_list_1[idx]["info"]["whole_arm_ratio"] > vlm_dict_list_2[idx]["info"]["whole_arm_ratio"]:
            pref_labels1.append(0)
        elif vlm_dict_list_1[idx]["info"]["whole_arm_ratio"] < vlm_dict_list_2[idx]["info"]["whole_arm_ratio"]:
            pref_labels1.append(1)
        else:
            pref_labels1.append(-1)

        # if gt_labels[-1] != pref_labels1[-1]:
        #     print('timestep:', vlm_dict_list_1[idx]["time_steps"], vlm_dict_list_2[idx]["time_steps"])
        #     print('ratio:', vlm_dict_list_1[idx]["info"]["whole_arm_ratio"], vlm_dict_list_2[idx]["info"]["whole_arm_ratio"])
        #     print('diff', abs(vlm_dict_list_1[idx]["info"]["whole_arm_ratio"] - vlm_dict_list_2[idx]["info"]["whole_arm_ratio"]))

        #     print('gt:', gt_labels[-1], 'pred:', pref_labels1[-1])
        #     plt.imshow(combined_image)
        #     plt.show()

        dict1_dist_shoulder = gripper_shoulder_dist(vlm_dict_list_1[idx]['gripper_pos'], vlm_dict_list_1[idx]['line_points'])
        dict2_dist_shoulder = gripper_shoulder_dist(vlm_dict_list_2[idx]['gripper_pos'], vlm_dict_list_2[idx]['line_points'])

        if dict1_dist_shoulder < dict2_dist_shoulder:
            pref_labels2.append(0)
        elif dict1_dist_shoulder > dict2_dist_shoulder:
            pref_labels2.append(1)
        else:
            pref_labels2.append(-1)
        

        dict1_dist_arm = gripper_arm_dist(vlm_dict_list_1[idx]['gripper_pos'], vlm_dict_list_1[idx]['line_points'])
        dict2_dist_arm = gripper_arm_dist(vlm_dict_list_2[idx]['gripper_pos'], vlm_dict_list_2[idx]['line_points'])

        if dict1_dist_arm < dict2_dist_arm:
            pref_labels3.append(0)
        elif dict1_dist_arm > dict2_dist_arm:
            pref_labels3.append(1)
        else:
            pref_labels3.append(-1)

        
    
    pref_labels1 = np.array(pref_labels1).reshape(-1, 1)
    # print('pref labels 1 len', pref_labels1.shape)

    good_idx1 = (pref_labels1 != -1).flatten()
    # print('good idx len', len(good_idx1))
    dict_list_1_l1 = []
    dict_list_2_l1 = []
    labels1 = []
    combined_images_list1 = []
    gt_labels_1 = []
    # print("vlm_labels:", vlm_labels)
    # print("good_idx:", good_idx)
    for ind in range(len(good_idx1)):
        if good_idx1[ind]:
            dict_list_1_l1.append(vlm_dict_list_1[ind])
            dict_list_2_l1.append(vlm_dict_list_2[ind])
            labels1.append(pref_labels1[ind])
            gt_labels_1.append(gt_labels[ind])
            # print("vlm_label:", vlm_labels[ind])
            combined_images_list1.append(combined_images_list[ind])
    # print('labels 1 len', len(labels1))

    pref_labels2 = np.array(pref_labels2).reshape(-1, 1)
    good_idx2 = (pref_labels2 != -1).flatten()
    dict_list_1_l2 = []
    dict_list_2_l2 = []
    labels2 = []
    combined_images_list2 = []
    gt_labels_2 = []

    for ind in range(len(good_idx2)):
        if good_idx2[ind]:
            dict_list_1_l2.append(vlm_dict_list_1[ind])
            dict_list_2_l2.append(vlm_dict_list_2[ind])
            labels2.append(pref_labels2[ind])
            gt_labels_2.append(gt_labels[ind])
            # print("vlm_label:", vlm_labels[ind])
            combined_images_list2.append(combined_images_list[ind])

    pref_labels3 = np.array(pref_labels3).reshape(-1, 1)
    good_idx3 = (pref_labels3 != -1).flatten()
    dict_list_1_l3 = []
    dict_list_2_l3 = []

    labels3 = []
    combined_images_list3 = []
    gt_labels_3 = []

    for ind in range(len(good_idx3)):
        if good_idx3[ind]:
            dict_list_1_l3.append(vlm_dict_list_1[ind])
            dict_list_2_l3.append(vlm_dict_list_2[ind])
            labels3.append(pref_labels3[ind])
            gt_labels_3.append(gt_labels[ind])
            # print("vlm_label:", vlm_labels[ind])
            combined_images_list3.append(combined_images_list[ind])

                
    if len(labels1) > 0:
        # print("label 1 obtained")
        # print("label:", labels1)
        # print("gt label:", gt_labels_1)
        from sklearn.metrics import classification_report, accuracy_score
        # print(classification_report(gt_labels_1, labels1))
        acc1 = accuracy_score(gt_labels_1, labels1)
        print('acc1:', acc1)

    if len(labels2) > 0:
        # print("label 2 obtained")
        # print("label:", labels2)
        # print("gt label:", gt_labels_2)
        from sklearn.metrics import classification_report, accuracy_score
        # print(classification_report(gt_labels_2, labels2))
        acc2 = accuracy_score(gt_labels_2, labels2)
        print('acc2:', acc2)

    if len(labels3) > 0:
        # print("label 2 obtained")
        # print("label:", labels2)
        # print("gt label:", gt_labels_2)
        from sklearn.metrics import classification_report, accuracy_score
        # print(classification_report(gt_labels_2, labels2))
        acc3 = accuracy_score(gt_labels_3, labels3)
        print('acc3:', acc3)
    
    labels1 = np.array(labels1).reshape(-1, 1)
    gt_labels_1 = np.array(gt_labels_1).reshape(-1, 1)
    labels2 = np.array(labels2).reshape(-1, 1)
    gt_labels_2 = np.array(gt_labels_2).reshape(-1, 1)
    labels3 = np.array(labels3).reshape(-1, 1)
    gt_labels_3 = np.array(gt_labels_3).reshape(-1, 1)
    save_path = '/scratch/alexis/data/traj_data_with_force_one-hot/labeled'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open("{}/{}_{}.pkl".format(save_path, 'labels1', batch_number), "wb") as f:
        pkl.dump([combined_images_list1, labels1, dict_list_1_l1, dict_list_2_l1, gt_labels_1], f, protocol=pkl.HIGHEST_PROTOCOL)

    with open("{}/{}_{}.pkl".format(save_path, 'labels2', batch_number), "wb") as f:
        pkl.dump([combined_images_list2, labels2, dict_list_1_l2, dict_list_2_l2, gt_labels_2], f, protocol=pkl.HIGHEST_PROTOCOL)

    with open("{}/{}_{}.pkl".format(save_path, 'labels3', batch_number), "wb") as f:
        pkl.dump([combined_images_list3, labels3, dict_list_1_l3, dict_list_2_l3, gt_labels_3], f, protocol=pkl.HIGHEST_PROTOCOL)
    return acc1, acc2, acc3

    # print("labels shape:", labels.shape)
    # print("obs1 shape:", len(obs1))
    # return labels1, labels2, labels3, dict_list_1, dict_list_2, combined_images_list1, combined_images_list2, combined_images_list3, gt_labels_1, gt_labels_2, gt_labels_3, acc1, acc2, acc3

def get_vlm_label(vlm_dict_list_1, vlm_dict_list_2):
    max_len = len(vlm_dict_list_1)
    print("Max len: ", max_len)
    batch_size = 64
    
    num_epochs = int(np.ceil(max_len/batch_size))
    print("Num epochs: ", num_epochs)
    acc1s = []
    acc2s = []
    acc3s = []
    for epoch in range(num_epochs):
        print('------epoch', epoch)
        start_index = epoch*batch_size        
        last_index = (epoch+1)*batch_size
        if last_index > max_len:
            last_index = max_len
        dict_list_1 = copy.deepcopy(vlm_dict_list_1[start_index:last_index])
        dict_list_2 = copy.deepcopy(vlm_dict_list_2[start_index:last_index])
        acc1, acc2, acc3 = get_label(dict_list_1, dict_list_2, epoch)

        # labels1, labels2, labels3, dict_list_1, dict_list_2, combined_images_list1, combined_images_list2, combined_images_list3, gt_labels_1, gt_labels_2, gt_labels_3, acc1, acc2, acc3 = get_label(dict_list_1, dict_list_2, epoch)
        acc1s.append(acc1)
        acc2s.append(acc2)
        acc3s.append(acc3)
    print(acc1s)
    print(acc2s)
    print(acc3s)
    print(sum(acc1s)/num_epochs, sum(acc2s)/num_epochs, sum(acc3s)/num_epochs)

def sample_v_dict(trajs, batch_size=4000):
    v_dict_list_1 = []
    v_dict_list_2 = []
    
    no_of_trajs = len(trajs)
    
    sample_batch = np.random.choice(no_of_trajs, size=batch_size, replace=True)
    print("Sample batch: ", sample_batch)
    for i in sample_batch:
        print("Trajectory: ", i)
        traj = trajs[i]
        traj_len = len(traj)
        print("Trajectory length: ", traj_len)

        while True:
            sample_index = np.random.choice(traj_len, size=2, replace=False)
            if abs(sample_index[0] - sample_index[1]) > 10:
                break
        print("Sample index: ", sample_index)
        dict_1 = {}
        dict_2 = {}
        dict_1["trajectory_index"] = i
        dict_2["trajectory_index"] = i

        dict_1["time_steps"] = sample_index[0]
        dict_2["time_steps"] = sample_index[1]

        
        dict_1["action"] = traj[sample_index[0]]["action"]
        dict_2["action"] = traj[sample_index[1]]["action"]
        
        dict_1["obs"] = traj[sample_index[0]]["obs"]
        dict_2["obs"] = traj[sample_index[1]]["obs"]
        
        dict_1["new_obs"] = traj[sample_index[0]]["new_obs"]
        dict_2["new_obs"] = traj[sample_index[1]]["new_obs"]

        dict_1["img"] = traj[sample_index[0]]["img"]
        dict_2["img"] = traj[sample_index[1]]["img"]

        dict_1["info"] = traj[sample_index[0]]["info"]
        dict_2["info"] = traj[sample_index[1]]["info"]


        dict_1["gripper_pos"] = traj[sample_index[0]]["gripper_pos"]
        dict_2["gripper_pos"] = traj[sample_index[1]]["gripper_pos"]

        dict_1["line_points"] = traj[sample_index[0]]["line_points"]
        dict_2["line_points"] = traj[sample_index[1]]["line_points"]

        dict_1["total_force"] = traj[sample_index[0]]["total_force"]
        dict_2["total_force"] = traj[sample_index[1]]["total_force"]

        
        v_dict_list_1.append(dict_1)
        v_dict_list_2.append(dict_2)
    
   
    return v_dict_list_1, v_dict_list_2

def train_reward_model(data_dir="/scratch/alexis/data/traj_data_with_force_one-hot/trajs"): 
    
    
    utils.set_seed_everywhere(0)
    # make directory for logging
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)    
 
    # create replay buffer
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    data = {}
    
    file_nos = os.listdir(data_dir)
    # file_nos = [file for file in file_nos if not file.endswith("0.0_0.0.pkl")]

    # file_nos.remove("videos")
    file_nos = sorted(file_nos)
    file_nos = [os.path.join(data_dir, file_no) for file_no in file_nos]
    print(len(file_nos))
    trajs = []
    for i in tqdm(range(len(file_nos))):
        print("Loading file ", file_nos[i])
        file = file_nos[i]
        with open(file, 'rb') as f:
            data = pkl.load(f)
        trajs.append(data)
        
    print("Total number of trajectories: ", len(trajs))
    v_dict_list_1 , v_dict_list_2 = sample_v_dict(trajs)
    get_vlm_label(v_dict_list_1, v_dict_list_2)
    
train_reward_model()
    
# exp_prefix = "vlm_cached_gpt4" #'2024-0728-real-model-experiments'
# load_variant_path = "/home/sreyas/Desktop/Projects/softagent_rpad/dressing_motion/curl/variant_store_cached_queries.json"
# loaded_vg = create_vg_from_json(load_variant_path)
# print("Loaded configs from ", load_variant_path)
# vg = loaded_vg
# print(vg)
# print('Number of configurations: ', len(vg.variants()))
# now = datetime.datetime.now(dateutil.tz.tzlocal())

# exp_count = 0
# timestamp = now.strftime('%m_%d_%H_%M_%S')
# exp_name = "real_world_vlm_queried-gpt4o-descriptive-prompt"
# print(exp_name)
# exp_name = "{}-{}-{:03}".format(exp_name, timestamp, exp_count)
# log_dir = "data" + "/local/" + exp_prefix + "/" + exp_name

# run_task(vg, log_dir=log_dir, exp_name=exp_name)