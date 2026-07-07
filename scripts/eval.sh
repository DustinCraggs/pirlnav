export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/il_objectnav.yaml"

DATA_DIR=$1
NV_DATASET=$2
PVR_DATASET=$3
EVAL_CHECKPOINT_DIR=$4
EXP_NAME=$5
GROUP_NAME=$6

DATA_PATH="$DATA_DIR/tasks/objectnav_hm3d_v1/"

set -x

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
    NUM_ENVIRONMENTS 20 \
    TEST_EPISODE_COUNT -1 \
    TASK_CONFIG.DATASET.TYPE "ObjectNav-v1" \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    EVAL.SPLIT "val" \
    # RL.DDPPO.force_distributed True \
    # EVAL.USE_CKPT_CONFIG False \
