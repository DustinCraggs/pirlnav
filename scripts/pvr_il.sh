set -x

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/il_objectnav.yaml"

DATA_DIR=$1
NV_DATASET=$2
PVR_DATASET=$3
EXP_NAME=$4
GROUP_NAME=$5

DATA_PATH="$DATA_DIR/demos/objectnav/objectnav_hm3d/objectnav_hm3d_hd"
TENSORBOARD_DIR="$DATA_DIR/tb/objectnav_il/$EXP_NAME/"
CHECKPOINT_DIR="$DATA_DIR/checkpoints/objectnav_il/$EXP_NAME/"
INFLECTION_COEF=3.234951275740812

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR

# MAIN_PORT=8740

echo "In ObjectNav IL DDP"
# python -u -m run \
    # --use_env \
    # --rdzv_endpoint localhost:29503 \
# --nnodes 1 \
python -u -m torch.distributed.run \
    --master_port 29504 \
    --nproc_per_node 1 \
    run.py \
    --exp-config $config \
    --run-type train \
    --seed 1000 \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    WB.GROUP $GROUP_NAME \
    WB.RUN_NAME $EXP_NAME \
    WB.MODE online \
    TRAINER_NAME "pvr-pirlnav-il" \
    NUM_ENVIRONMENTS 32 \
    NUM_UPDATES 52000 \
    IL.BehaviorCloning.wd 1e-6 \
    IL.BehaviorCloning.num_steps 64 \
    IL.BehaviorCloning.num_mini_batch 8 \
    IL.BehaviorCloning.use_gradient_accumulation True \
    IL.BehaviorCloning.num_accumulated_gradient_steps 8 \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
    POLICY.PVR_ENCODER.num_heads 4 \
    POLICY.PVR_ENCODER.num_layers 2 \
    POLICY.PVR_ENCODER.dropout 0.1 \
    POLICY.SEQ2SEQ.use_prev_action True \
    POLICY.SEQ2SEQ.use_final_obs_resid_mlp False \
    TASK_CONFIG.PVR.use_pvr_encoder False \
    POLICY.RGB_ENCODER.input_channels 4 \
    NUM_CHECKPOINTS -1 \
    CHECKPOINT_INTERVAL 5000 \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.PVR.non_visual_obs_data_path $NV_DATASET \
    TASK_CONFIG.PVR.pvr_data_path $PVR_DATASET \
    # POLICY.STATE_ENCODER.hidden_size 128 \
    # NUM_UPDATES 102000 \
    # NUM_ENVIRONMENTS 4 \
    # TASK_CONFIG.PVR.use_fixed_size_embedding True \


# MAIN_PORT=8741 CUDA_VISIBLE_DEVICES=0 ./scripts/pvr_il.sh data /storage/dc/pvr_data/twenty_percent/non_visual_data/ /storage/dc/pvr_data/twenty_percent/clip_data/ clip_20pc ten_percent
# MAIN_PORT=8742 CUDA_VISIBLE_DEVICES=1 ./scripts/pvr_il.sh data /storage/dc/pvr_data/twenty_five_percent/non_visual_data/ /storage/dc/pvr_data/twenty_five_percent/clip_data/ clip_25pc ten_percent

# MAIN_PORT=8741 CUDA_VISIBLE_DEVICES=0 ./scripts/pvr_il.sh data \
#     /storage/dc/pvr_data/stretch_like/twenty_percent/clip_non_visual_no_look_actions/ \
#     /storage/dc/pvr_data/stretch_like/twenty_percent/clip_non_visual_no_look_actions/ \
#     test \
#     test