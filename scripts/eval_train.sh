export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# dataset=$1

config="configs/experiments/il_objectnav.yaml"

DATA_DIR=$1
PVR_DIR=$2
EVAL_CHECKPOINT_DIR=$3
EXP_NAME=$4
GROUP_NAME=$5

DATA_PATH="$DATA_DIR/demos/objectnav/objectnav_hm3d/objectnav_hm3d_hd"

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
    NUM_ENVIRONMENTS 40 \
    TEST_EPISODE_COUNT -1 \
    TASK_CONFIG.DATASET.TYPE "ObjectNav-v2" \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    EVAL.SPLIT "train" \
    TASK_CONFIG.DATASET.SUB_SPLIT_INDEX_PATH "$PVR_DIR/ep_index.json" \

