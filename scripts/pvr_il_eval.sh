export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/il_objectnav.yaml"

DATA_DIR=$1
NV_DATASET=$2
PVR_DATASET=$3
PVR_KEY=$4
COSTMAP_CHANNELS=$5
EVAL_CHECKPOINT_DIR=$6
EXP_NAME=$7
GROUP_NAME=$8
INPUT_CHANNELS=$9
COST_PREDICTOR_PATH=${10}

# DATA_PATH="$DATA_DIR/demos/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_hd"
DATA_PATH="$DATA_DIR/tasks/objectnav_hm3d_v1/"

set -x

echo "In ObjectNav IL DDP"

# If SLURM_NTASKS > 1, it will use distributed mode which is not supported for eval:
unset SLURM_JOBID
unset SLURM_NTASKS

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
    NUM_ENVIRONMENTS 8 \
    EVAL.SPLIT "val" \
    EVAL.USE_CKPT_CONFIG False \
    TASK_CONFIG.DATASET.TYPE "ObjectNav-v1" \
    TASK_CONFIG.SIMULATOR.ACTION_SPACE_CONFIG "v1_no_op_look" \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    TASK_CONFIG.PVR.pvr_data_path $PVR_DATASET \
    TASK_CONFIG.PVR.non_visual_obs_data_path $NV_DATASET \
    POLICY.PVR_ENCODER.num_heads 4 \
    POLICY.PVR_ENCODER.num_layers 2 \
    POLICY.PVR_ENCODER.dropout 0.1 \
    POLICY.SEQ2SEQ.use_prev_action True \
    POLICY.SEQ2SEQ.use_final_obs_resid_mlp False \
    TASK_CONFIG.PVR.use_pvr_encoder False \
    POLICY.RGB_ENCODER.input_channels $INPUT_CHANNELS \
    POLICY.RGB_ENCODER.costmap_channels $COSTMAP_CHANNELS \
    POLICY.RGB_ENCODER.use_augmentations_test_time True \
    TASK_CONFIG.PVR.pvr_key $PVR_KEY \
    TASK_CONFIG.REPRESENTATION_GENERATOR.data_generators.predicted_costmap.cost_predictor_checkpoint_paths "['${COST_PREDICTOR_PATH}']" \
    TASK_CONFIG.REPRESENTATION_GENERATOR.data_generators.predicted_costmap.distributional_cost_predictors "[]"

    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.name clip \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.clip_kwargs.model_path None \
    # TASK_CONFIG.PVR.use_fixed_size_embedding True \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.clip_kwargs.model_path "/data/drive2/models/clip-vit-base-patch32" \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.name clip \
    # TASK_CONFIG.DATASET.SUB_SPLIT_INDEX_PATH "$DATA_DIR/pvr_demos/ten_percent/ep_index.json" \

    # POLICY.RGB_ENCODER.augmentations_name "" \
