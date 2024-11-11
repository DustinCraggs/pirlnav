#!/bin/bash
#SBATCH --job-name=pirlnav
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@1000
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --output=slurm_logs/ddpil-eval-%j.out
#SBATCH --error=slurm_logs/ddpil-eval-%j.err
#SBATCH --requeue

# source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
# conda deactivate
# conda activate pirlnav

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
# export MASTER_ADDR

# cd /srv/flash1/rramrakhya6/spring_2022/pirlnav

config="configs/experiments/il_objectnav.yaml"

# DATA_PATH="../data/habitat/demos/data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_hd/"
DATA_PATH="data/demos/objectnav_hm3d_v1/"
# DATA_PATH="data/demos/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_hd/"
# TENSORBOARD_DIR="../data/habitat/tb/objectnav_il/test/"
TENSORBOARD_DIR="data/habitat/tb/objectnav_il/test/"
EVAL_CHECKPOINT_DIR="data/checkpoints/objectnav_il/$1"
# EVAL_CHECKPOINT_DIR="../data/checkpoints/objectnav_il/$1"

# EVAL_CKPT_PATH_DIR=$1

mkdir -p $TENSORBOARD_DIR
set -x

echo "In ObjectNav IL eval"
python -u -m run \
    --exp-config $config \
    --run-type eval \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    EVAL_CKPT_PATH_DIR $EVAL_CHECKPOINT_DIR \
    VIDEO_DIR "video_dir/train/$1/" \
    WB.GROUP "pirlnav_eval" \
    WB.RUN_NAME eval_$1 \
    TEST_EPISODE_COUNT 32 \
    NUM_ENVIRONMENTS 1 \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.SPLIT "train" \
    EVAL.SPLIT "val" \
    EVAL.USE_CKPT_CONFIG False \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    TASK_CONFIG.DATASET.EPISODE_STRIDE 1 \
