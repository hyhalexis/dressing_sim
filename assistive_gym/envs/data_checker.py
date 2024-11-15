import numpy as np
import pickle5 as pickle

traj_path = 'traj_data/p1_motion4_front_tshirt_26_0'
with open (traj_path, 'rb') as f:
    traj_data = pickle.load(f)

print('Length: {}'.format(len(traj_data)))
print('First entry: {}'.format(traj_data[0]))
print('Last entry: {}'.format(traj_data[-1]))
print(traj_data[0]['obs'].pos)
