export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

DATA_DIR=$1
OUTPUT_JSON_PATH=$2
STRIDE=$3
START_IDX=$4

# BC_DATASET_PATH="$DATA_DIR/tasks/objectnav_hm3d_v1/{split}/{split}.json.gz"
BC_DATASET_PATH="$DATA_DIR/demos/objectnav/objectnav_hm3d/objectnav_hm3d_hd/{split}/{split}.json.gz"

CUDA_VISIBLE_DEVICES=1 python run.py \
    --run-type gen_sub_split \
    --exp-config configs/experiments/il_objectnav.yaml \
    TASK_CONFIG.DATASET.DATA_PATH $BC_DATASET_PATH \
    TASK_CONFIG.SUB_SPLIT_GENERATOR.INDEX_PATH $OUTPUT_JSON_PATH \
    TASK_CONFIG.SUB_SPLIT_GENERATOR.STRIDE $STRIDE \
    TASK_CONFIG.SUB_SPLIT_GENERATOR.START_IDX $START_IDX \
    # EVAL.SPLIT "val" \
    # TASK_CONFIG.DATASET.SPLIT "val" \
    # TASK_CONFIG.DATASET.TYPE "ObjectNav-v1" \

