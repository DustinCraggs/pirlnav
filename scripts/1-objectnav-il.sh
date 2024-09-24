export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# dataset=$1

config="configs/experiments/il_objectnav.yaml"

# DATA_PATH="data/datasets/objectnav/objectnav_hm3d/${dataset}"
# TENSORBOARD_DIR="tb/objectnav_il/${dataset}/ovrl_resnet50/seed_1/"
# CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/${dataset}/ovrl_resnet50/seed_1/"

DATA_PATH="../data/habitat/demos/data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_hd/"
TENSORBOARD_DIR="../data/habitat/tb/objectnav_il/test/"
CHECKPOINT_DIR="../data/checkpoints/objectnav_il/$1/"
INFLECTION_COEF=3.234951275740812

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR
set -x

echo "In ObjectNav IL DDP"
# python -u -m run \
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 2 \
    run.py \
    --exp-config $config \
    --run-type train \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 20000 \
    NUM_ENVIRONMENTS 16 \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
    IL.BehaviorCloning.num_mini_batch 8 \
    RL.DDPPO.force_distributed True \
    POLICY.RGB_ENCODER.pretrained_encoder data/visual_encoders/omnidata_DINO_02.pth \
    # RL.DDPPO.train_encoder False \
