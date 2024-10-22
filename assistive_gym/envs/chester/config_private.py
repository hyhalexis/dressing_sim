import os.path as osp
import os

PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))
LOG_DIR = os.path.join(PROJECT_PATH, "data")

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
}

REMOTE_LOG_DIR = {
    'seuss': '/data/{}/{}/data'.format(HOME_FOLDER, PROJECT_NAME),
    'autobot': '/project_data/held/{}/{}/data'.format(HOME_FOLDER, PROJECT_NAME),

}

# slurm header to write for the job
REMOTE_HEADER = dict(seuss="""
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=10
#SBATCH --time=480:00:00
#SBATCH --gres=gpu:1 
#SBATCH --mem=10G
""".strip(),
autobot="""
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --exclude=autobot-0-[25,29,33,37]
#SBATCH --time=3-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
""".strip())

# location of the singularity file related to the project
SIMG_DIR = {
    'seuss': '/home/yufeiw2/softgymcontainer_v3.simg',
    'autobot': '/home/xlin3/softgym_containers/softgymcontainer_v3.simg',

}
CUDA_MODULE = {
    'seuss': 'cuda-91',
    'autobot': 'cuda-11.1.1',
    'psc': 'cuda/9.0',
}
MODULES = {
    'seuss': ['singularity'],
    'autobot': ['singularity'],
    'psc': ['singularity'],
}


AUTOBOT_NODELIST = [
    '1-1', '1-6',  # 10x2080 + 40 cpus
    '0-9', '0-11', '0-13', '0-15', '0-17', '0-19', '0-21', '0-23',  # 4x2080 + 32 cpus
    '0-25', '0-29', '0-33', '0-37',  # 8x3090 + 96 cpus
]
GPU_STATE_DIR = '/project_data/mtcmon/nodestats/'
CHESTER_QUEUE_DIR = '/home/yufeiw2/chester_scheduler/queues'
CHESTER_CHEDULER_LOG_DIR = '/home/yufeiw2/chester_scheduler/logs/'

DISABLE_SINGULARITY = False
