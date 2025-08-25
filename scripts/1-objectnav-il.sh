export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# dataset=$1

config="configs/experiments/il_objectnav.yaml"

# DATA_PATH="data/datasets/objectnav/objectnav_hm3d/${dataset}"
# TENSORBOARD_DIR="tb/objectnav_il/${dataset}/ovrl_resnet50/seed_1/"
# CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/${dataset}/ovrl_resnet50/seed_1/"

DATA_DIR=$1
EXP_NAME=$2
GROUP_NAME=$3

DATA_PATH="$1/demos/objectnav/objectnav_hm3d/objectnav_hm3d_hd"
TENSORBOARD_DIR="$1/tb/objectnav_il/$2/"
CHECKPOINT_DIR="$1/checkpoints/objectnav_il/$2/"
INFLECTION_COEF=3.234951275740812

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR
set -x

echo "In ObjectNav IL DDP"
# python -u -m run \
# python -u -m torch.distributed.launch \
    # --use_env \
python -u -m torch.distributed.run \
    --master_port 29502 \
    --nproc_per_node 1 \
    run.py \
    --exp-config $config \
    --run-type train \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 52000 \
    NUM_ENVIRONMENTS 16 \
    IL.BehaviorCloning.num_steps 64 \
    IL.BehaviorCloning.num_mini_batch 16 \
    IL.BehaviorCloning.use_gradient_accumulation True \
    IL.BehaviorCloning.num_accumulated_gradient_steps 8 \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
    WB.GROUP $GROUP_NAME \
    WB.RUN_NAME $EXP_NAME \
    WB.MODE online \
    POLICY.RGB_ENCODER.pretrained_encoder $DATA_DIR/visual_encoders/omnidata_DINO_02.pth \
    NUM_CHECKPOINTS -1 \
    CHECKPOINT_INTERVAL 2500 \
    POLICY.RGB_ENCODER.normalize_visual_inputs True \
    TASK_CONFIG.DATASET.SUB_SPLIT_INDEX_PATH /storage/dc/pvr_data/ten_percent/ep_index.json \
    # TASK_CONFIG.DATASET.SUB_SPLIT_INDEX_PATH "temp/ep_index.json" \
