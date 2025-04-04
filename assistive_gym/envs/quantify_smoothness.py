import numpy as np
import os
import pickle

def compute_smoothness_metrics(actions):
    velocities = actions  # Velocity is just the action vector
    accelerations = np.diff(velocities, axis=0)  # First derivative of velocity
    jerks = np.diff(accelerations, axis=0)  # First derivative of acceleration

    metrics = {
        "mean_velocity": np.mean(np.linalg.norm(velocities, axis=1)),
        # "std_velocity": np.std(np.linalg.norm(velocities, axis=1)),
        "mean_acceleration": np.mean(np.linalg.norm(accelerations, axis=1)),
        # "std_acceleration": np.std(np.linalg.norm(accelerations, axis=1)),
        "mean_jerk": np.mean(np.linalg.norm(jerks, axis=1)),
        # "std_jerk": np.std(np.linalg.norm(jerks, axis=1)),
        "length": actions.shape[0]
    }
    return metrics


path = '/project_data/held/ahao/data/2025-0313-pybullet-eval-baselines/eval-all-baseline+ours-0.01_thresh_for_fcvp_high-fric-unseen1-03_13_08_24_53-000/trajs'
pkl_files = sorted([f for f in os.listdir(path)])
policy_metrics = {"p0": [], "p1": [], "p2": [], "p3": [], "p4": [], "p5": []}


for i, filename in enumerate(pkl_files):
    file_path = os.path.join(path, filename)
    policy = filename[:2]
    with open(file_path, 'rb') as f:
        print(filename)
        data = pickle.load(f)
        actions = []
        for d in data:
            actions.append(d['step_action'])

        actions = np.array(actions)

    # Example usage
    smoothness_metrics = compute_smoothness_metrics(actions)
    # print(smoothness_metrics)
    policy_metrics[policy].append(smoothness_metrics)

# Aggregate results (compute mean and std for each metric per policy)
summary_stats = {}
for policy, metrics_list in policy_metrics.items():
    if metrics_list:
        keys = metrics_list[0].keys()
        summary_stats[policy] = {
            key: (np.mean([m[key] for m in metrics_list]), np.std([m[key] for m in metrics_list]))
            for key in keys
        }

# Print results
for policy, stats in summary_stats.items():
    print(f"Policy {policy}:")
    for metric, (mean, std) in stats.items():
        print(f"  {metric}: {mean:.4f} ± {std:.4f}")
    print()
