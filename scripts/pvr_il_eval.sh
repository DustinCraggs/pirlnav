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

# DATA_PATH="$DATA_DIR/demos/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_hd"
DATA_PATH="$DATA_DIR/tasks/objectnav_hm3d_v1/"

set -x

echo "In ObjectNav IL DDP"

python -u -m run \
    --exp-config $config \
    --run-type eval \
    --seed 1000 \
    EVAL_CKPT_PATH_DIR $EVAL_CHECKPOINT_DIR \
    WB.PROJECT_NAME habitat-bc-eval \
    WB.GROUP $GROUP_NAME \
    WB.RUN_NAME $EXP_NAME \
    WB.MODE online \
    VIDEO_DIR "$DATA_DIR/videos/$GROUP_NAME/$EXP_NAME" \
    TRAINER_NAME "pvr-pirlnav-il" \
    TEST_EPISODE_COUNT -1 \
    NUM_ENVIRONMENTS 10 \
    EVAL.SPLIT "val" \
    EVAL.USE_CKPT_CONFIG False \
    TASK_CONFIG.DATASET.TYPE "ObjectNav-v1" \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    TASK_CONFIG.PVR.pvr_data_path $PVR_DATASET \
    TASK_CONFIG.PVR.non_visual_obs_data_path $NV_DATASET \
    POLICY.PVR_ENCODER.num_heads 4 \
    POLICY.PVR_ENCODER.num_layers 2 \
    POLICY.PVR_ENCODER.dropout 0.1 \
    POLICY.SEQ2SEQ.use_prev_action True \
    POLICY.SEQ2SEQ.use_final_obs_resid_mlp False \
    TASK_CONFIG.PVR.use_pvr_encoder False \
    POLICY.RGB_ENCODER.input_channels 4 \
    POLICY.RGB_ENCODER.use_augmentations_test_time True \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.name clip \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.clip_kwargs.model_path None \
    # TASK_CONFIG.PVR.use_fixed_size_embedding True \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.clip_kwargs.model_path "/data/drive2/models/clip-vit-base-patch32" \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.name clip \
    # TASK_CONFIG.DATASET.SUB_SPLIT_INDEX_PATH "$DATA_DIR/pvr_demos/ten_percent/ep_index.json" \

    # POLICY.RGB_ENCODER.augmentations_name "" \

# MAIN_PORT=8740 CUDA_VISIBLE_DEVICES=1 ./scripts/eval_train.sh data /storage/dc/pvr_data/one_ep ../data/checkpoints/objectnav_il/pirlnav_test/ckpt.0.pth pirlnav_il_1pc_3_ckpt_2_train one_percent_eval
# MAIN_PORT=8741 CUDA_VISIBLE_DEVICES=1 ./scripts/pvr_il_eval_train.sh data /storage/dc/pvr_data/ten_percent/ ../data/checkpoints/objectnav_il/pvr_clip_ten_percent_heads_4_long_4/ckpt.2.pth test_video_2 test_video_2
# MAIN_PORT=8741 CUDA_VISIBLE_DEVICES=1 ./scripts/pvr_il_eval_train.sh data /storage/dc/pvr_data/stretch_like/twenty_percent/clip_non_visual_movement_only/ /storage/dc/pvr_data/stretch_like/twenty_percent/clip_non_visual_movement_only/ data/checkpoints/objectnav_il/clip_no_pose_3/ckpt.19.pth stretch_20pc clip_no_gps_no_compass