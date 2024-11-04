export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

data_dir=$1
output_json_path=$2

bc_dataset_path="$1/demos/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_hd/{split}/{split}.json.gz"

CUDA_VISIBLE_DEVICES=1 python run.py \
    --run-type gen_sub_split \
    --exp-config configs/experiments/il_objectnav.yaml \
    TASK_CONFIG.DATASET.DATA_PATH $bc_dataset_path \
    TASK_CONFIG.SUB_SPLIT_GENERATOR.INDEX_PATH $output_json_path \
    TASK_CONFIG.SUB_SPLIT_GENERATOR.STRIDE 10 \

