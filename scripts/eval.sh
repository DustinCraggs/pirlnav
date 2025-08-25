export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# dataset=$1

config="configs/experiments/il_objectnav.yaml"

# DATA_PATH="data/datasets/objectnav/objectnav_hm3d/${dataset}"
# TENSORBOARD_DIR="tb/objectnav_il/${dataset}/ovrl_resnet50/seed_1/"
# CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/${dataset}/ovrl_resnet50/seed_1/"

DATA_DIR=$1
EVAL_CHECKPOINT_DIR=$2
EXP_NAME=$3
GROUP_NAME=$4

DATA_PATH="$DATA_DIR/tasks/objectnav_hm3d_v1/"

# DATA_PATH="../data/habitat/demos/data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/"
# TENSORBOARD_DIR="../data/habitat/tb/objectnav_il/test/"
# CHECKPOINT_DIR="../data/checkpoints/objectnav_il/$1/"

mkdir -p $TENSORBOARD_DIR
set -x

echo "In ObjectNav IL eval"
python -u -m run \
    --exp-config $config \
    --run-type eval \
    EVAL_CKPT_PATH_DIR $EVAL_CHECKPOINT_DIR \
    VIDEO_DIR "$DATA_DIR/videos/$EXP_NAME/" \
    WB.GROUP $GROUP_NAME \
    WB.RUN_NAME $EXP_NAME \
    WB.MODE online \
    NUM_ENVIRONMENTS 20 \
    TEST_EPISODE_COUNT -1 \
    TASK_CONFIG.DATASET.TYPE "ObjectNav-v1" \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    EVAL.SPLIT "val" \
    # RL.DDPPO.force_distributed True \
    # EVAL.USE_CKPT_CONFIG False \
