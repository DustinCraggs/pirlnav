export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# dataset=$1

config="configs/experiments/il_objectnav.yaml"


DATA_DIR=$1
EXP_NAME=$2

DATA_PATH="$DATA_DIR/demos/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_hd"
TENSORBOARD_DIR="$DATA_DIR/tb/objectnav_il/$EXP_NAME/"
CHECKPOINT_DIR="$DATA_DIR/checkpoints/objectnav_il/$EXP_NAME/"
INFLECTION_COEF=3.234951275740812

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR
set -x

echo "In ObjectNav IL DDP"
# python -u -m run \
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 1 \
    run.py \
    --exp-config $config \
    --run-type train \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    WB.RUN_NAME $EXP_NAME \
    TRAINER_NAME "pvr-pirlnav-il" \
    NUM_UPDATES 391000 \
    NUM_ENVIRONMENTS 1 \
    IL.BehaviorCloning.num_steps 1024 \
    IL.BehaviorCloning.use_gradient_accumulation True \
    IL.BehaviorCloning.num_accumulated_gradient_steps 32 \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
    IL.BehaviorCloning.num_mini_batch 1 \
    POLICY.PVR_ENCODER.num_heads 4 \
    NUM_CHECKPOINTS -1 \
    CHECKPOINT_INTERVAL 5000 \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.PVR.pvr_data_path "$DATA_DIR/pvr_demos/ten_percent/clip_data" \
    TASK_CONFIG.PVR.non_visual_obs_data_path "$DATA_DIR/pvr_demos/ten_percent/non_visual_data"
