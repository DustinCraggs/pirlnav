export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/il_objectnav.yaml"

DATA_DIR=$1
PVR_DIR=$2
EVAL_CHECKPOINT_DIR=$3
EXP_NAME=$4

DATA_PATH="$DATA_DIR/demos/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_hd"

set -x

echo "In ObjectNav IL DDP"

python -u -m run \
    --exp-config $config \
    --run-type eval \
    EVAL_CKPT_PATH_DIR $EVAL_CHECKPOINT_DIR \
    VIDEO_DIR "$DATA_DIR/videos/$EXP_NAME/" \
    WB.GROUP "pvr_eval" \
    WB.RUN_NAME $EXP_NAME \
    WB.MODE online \
    TRAINER_NAME "pvr-pirlnav-il" \
    TEST_EPISODE_COUNT -1 \
    NUM_ENVIRONMENTS 20 \
    EVAL.SPLIT "train" \
    TASK_CONFIG.DATASET.TYPE "ObjectNav-v2" \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    TASK_CONFIG.PVR.pvr_data_path "$PVR_DIR/one_percent/clip_data" \
    TASK_CONFIG.PVR.non_visual_obs_data_path "$PVR_DIR/one_percent/non_visual_data" \
    TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.name clip \
    TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.clip_kwargs.model_path "/data/drive2/models/clip-vit-base-patch32" \
    TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.clip_kwargs.use_float16 True \
    POLICY.PVR_ENCODER.num_heads 4 \
    TASK_CONFIG.DATASET.SUB_SPLIT_INDEX_PATH "$DATA_DIR/pvr_demos/ten_percent/ep_index.json" \
