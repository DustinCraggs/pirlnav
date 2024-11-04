export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# dataset=$1

config="configs/experiments/il_objectnav.yaml"

# DATA_PATH="../data/habitat/demos/data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_hd/"
DATA_PATH="../data/habitat/demos/data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/"
TENSORBOARD_DIR="../data/habitat/tb/objectnav_il/test/"
EVAL_CHECKPOINT_DIR="../data/checkpoints/objectnav_il/$1"

mkdir -p $TENSORBOARD_DIR
set -x

echo "In ObjectNav IL DDP"

python -u -m run \
    --exp-config $config \
    --run-type eval \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    EVAL_CKPT_PATH_DIR $EVAL_CHECKPOINT_DIR \
    VIDEO_DIR "video_dir/train/$1/" \
    WB.RUN_NAME eval_$1 \
    TRAINER_NAME "pvr-pirlnav-il" \
    TEST_EPISODE_COUNT 32 \
    NUM_ENVIRONMENTS 4 \
    TASK_CONFIG.DATASET.SPLIT "val" \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    POLICY.PVR_ENCODER.num_heads 4