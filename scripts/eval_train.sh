export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# dataset=$1

config="configs/experiments/il_objectnav.yaml"

DATA_DIR=$1
NV_DATASET=$2
PVR_DATASET=$3
EVAL_CHECKPOINT_DIR=$4
EXP_NAME=$5
GROUP_NAME=$6
SUB_SPLIT_INDEX_PATH=$7

DATA_PATH="$DATA_DIR/demos/objectnav/objectnav_hm3d/objectnav_hm3d_hd"

set -x

echo $PVR_DIR

python -u -m run \
    --exp-config $config \
    --run-type eval \
    TRAINER_NAME "pvr-pirlnav-il" \
    EVAL_CKPT_PATH_DIR $EVAL_CHECKPOINT_DIR \
    TASK_CONFIG.PVR.non_visual_obs_data_path $NV_DATASET \
    TASK_CONFIG.PVR.pvr_data_path $PVR_DATASET \
    VIDEO_DIR "$DATA_DIR/videos/$EXP_NAME/" \
    WB.PROJECT_NAME habitat-bc-eval \
    WB.GROUP $GROUP_NAME \
    WB.RUN_NAME $EXP_NAME \
    WB.MODE online \
    NUM_ENVIRONMENTS 1 \
    TEST_EPISODE_COUNT -1 \
    TASK_CONFIG.DATASET.TYPE "ObjectNav-v2" \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    EVAL.SPLIT "train" \
    TASK_CONFIG.DATASET.SUB_SPLIT_INDEX_PATH $SUB_SPLIT_INDEX_PATH \

