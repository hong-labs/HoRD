#!/bin/bash
#SBATCH --job-name=hord-<YOUR_USERNAME>
#SBATCH --time=48:00:00
#SBATCH --open-mode=append
#SBATCH --output=<YOUR_DATA_ROOT>/workspace/hord/sbatch_logs/%x-%j.out
#SBATCH --error=<YOUR_DATA_ROOT>/workspace/hord/sbatch_logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>


# Conda
source <YOUR_DATA_ROOT>/miniconda3/etc/profile.d/conda.sh
conda activate hord

# Isaac Sim
export ISAACSIM_PATH="<YOUR_DATA_ROOT>/isaacsim"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

# Avoid $HOME writes
export XDG_DATA_HOME=<YOUR_DATA_ROOT>/.local/share
export XDG_CACHE_HOME=<YOUR_DATA_ROOT>/.cache
export XDG_CONFIG_HOME=<YOUR_DATA_ROOT>/.config

export CUDA_VISIBLE_DEVICES=7
export HYDRA_FULL_ERROR=1

# IsaacLab
source <YOUR_DATA_ROOT>/workspace/IsaacLab/_isaac_sim/setup_conda_env.sh

echo $CUDA_VISIBLE_DEVICES
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.current_device())"
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Run training
cd <YOUR_DATA_ROOT>/workspace/hord/
python hord/train_agent.py +exp=full_body_tracker/transformer +robot=g1  +simulator=isaaclab  motion_file=data/yaml_files/train_g1_long.pt +experiment_name=full_body_tracker_g1_noDR ++headless=True
