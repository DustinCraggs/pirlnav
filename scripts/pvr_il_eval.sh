export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# dataset=$1

config="configs/experiments/il_objectnav.yaml"

DATA_PATH="../data/habitat/demos/data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/"
TENSORBOARD_DIR="../data/habitat/tb/objectnav_il/test/"
EVAL_CHECKPOINT_DIR="../data/checkpoints/objectnav_il/$1"
INFLECTION_COEF=3.234951275740812

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR
set -x

echo "In ObjectNav IL DDP"

python -u -m run \
    --exp-config $config \
    --run-type eval \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    EVAL_CKPT_PATH_DIR $EVAL_CHECKPOINT_DIR \
    VIDEO_DIR "video_dir/$1/" \
    WB.RUN_NAME eval_$1 \
    TRAINER_NAME "pvr-pirlnav-il" \
    TEST_EPISODE_COUNT 10 \
    NUM_ENVIRONMENTS 10 \
    IL.BehaviorCloning.num_steps 256 \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
    IL.BehaviorCloning.num_mini_batch 1 \
    POLICY.PVR_ENCODER.num_heads 12 \
    RL.DDPPO.force_distributed True
