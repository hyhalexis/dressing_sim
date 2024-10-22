import os.path as osp
import os

# TODO change this before make it into a pip package
PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))

LOG_DIR = os.path.join(PROJECT_PATH, "data")

# Make sure to use absolute path
HOME_FOLDER = 'yufeiw2'
PROJECT_NAME = 'softagent_dressing'

# Make sure to use absolute path
REMOTE_DIR = {
    'seuss': '/home/{}/Projects/{}'.format(HOME_FOLDER, PROJECT_NAME),
    'autobot': '/home/{}/Projects/{}'.format(HOME_FOLDER, PROJECT_NAME),

}

REMOTE_MOUNT_OPTION = {
    'seuss': '/usr/share/glvnd',
    'autobot': '/opt/',
    # 'psc': '/pylon5/ir5fpfp/xlin3/Projects/baselines_hrl/:/mnt',
}

REMOTE_LOG_DIR = {
    # 'seuss': os.path.join(REMOTE_DIR['seuss'], "data"),
    'seuss': '/data/yufeiw2/softagent_dressing/data',
    'autobot': '/project_data/held/{}/{}/data'.format(HOME_FOLDER, PROJECT_NAME),
    # 'psc': os.path.join(REMOTE_DIR['psc'], "data")
    'psc': os.path.join('/mnt', "data"),
}

# PSC: https://www.psc.edu/bridges/user-guide/running-jobs
# partition include [RM, RM-shared, LM, GPU]
# TODO change cpu-per-task based on the actual cpus needed (on psc)
# #SBATCH --exclude=compute-0-[7,11]
# Adding this will make the job to grab the whole gpu. #SBATCH --gres=gpu:1
#SBATCH --exclude=compute-0-[5]
#SBATCH --exclude=compute-0-17,compute-0-19,compute-0-21,compute-0-23,compute-0-25,compute-0-27
#SBATCH --exclude=compute-0-17,compute-0-19,compute-0-21,compute-0-23,compute-0-25,compute-0-27,compute-1-9
REMOTE_HEADER = dict(seuss="""
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --exclude=compute-0-17
#SBATCH --cpus-per-task=6
#SBATCH --time=480:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
""".strip(), psc="""
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=RM
#SBATCH --ntasks-per-node=18
#SBATCH --time=48:00:00
#SBATCH --mem=64G
""".strip(), psc_gpu="""
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared  
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=48:00:00
""".strip(), autobot="""
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --exclude=autobot-0-[25,29,33,37]
#SBATCH --time=3-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
""".strip())

#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=11
#SBATCH --time=7-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=autobot-1-[1,6]
#SBATCH --mem=80G

#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --time=3-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=autobot-1-[1,6]
#SBATCH --mem=40G



# location of the singularity file related to the project
SIMG_DIR = {
    'seuss': '/home/zixuanhu/containers/softgymcontainer_v3.simg',
    'autobot': '/home/xlin3/softgym_containers/softgymcontainer_v3.simg',
    # 'psc': '$SCRATCH/containers/ubuntu-16.04-lts-rl.img',
    'psc': '/pylon5/ir5fpfp/xlin3/containers/ubuntu-16.04-lts-rl.img',

}
CUDA_MODULE = {
    'seuss': 'cuda-91',
    # 'autobot': 'cuda-11.1.1',
    'autobot': 'cuda-10.2',
    'psc': 'cuda/9.0',
}
MODULES = {
    'seuss': ['singularity'],
    'autobot': ['singularity'],
    'psc': ['singularity'],
}


AUTOBOT_NODELIST = [
    # '0-9', 
    # '0-11', 
    # '0-13', 
    # '0-15', 
    # '0-17', 
    # '0-19', 
    # '0-21',
    # '0-23',    
      # 4x2080 + 32 cpus
    # '0-25', 
    # '0-29', 
    '0-33', 
    # '0-37',  # 8x3090 + 96 cpus
    # '1-10', '1-14' # 10 x 3080 ti
    # '1-1', 
    # '1-6',   # 10x2080 + 40 cpus

]
GPU_STATE_DIR = '/project_data/mtcmon/nodestats/'
CHESTER_QUEUE_DIR = '/home/yufeiw2/chester_scheduler/queues'
CHESTER_CHEDULER_LOG_DIR = '/home/yufeiw2/chester_scheduler/logs/'

DISABLE_SINGULARITY = True