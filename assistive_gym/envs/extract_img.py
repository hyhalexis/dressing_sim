import numpy as np
import pickle5 as pickle
import matplotlib.pyplot as plt
import os

traj_path = '/home/alexis/2024_12_17_20_56_51' # traj folder path

files = os.listdir(traj_path)
def extract_number(filename):
    try:
        return int(filename.split('_')[-1].split('.')[0])
    except ValueError:
        return None

files = sorted(
    [f for f in os.listdir(traj_path) if f.endswith('.pkl') and extract_number(f) is not None],
    key=lambda x: extract_number(x)
)

imgs = []
for file in files:
    print(file)
    with open (os.path.join(traj_path, file), 'rb') as f:
        traj_data = pickle.load(f, encoding='latin1')

    image = traj_data['image']
    image = np.rot90(image, k=1)
    imgs.append(image)

output_file = '/home/alexis/processed_images.pkl' # img list saved path
with open(output_file, 'wb') as f:
    pickle.dump(imgs, f)
print(f"All images have been saved to {output_file}")
