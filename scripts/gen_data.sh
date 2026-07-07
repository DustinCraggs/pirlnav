export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
# export GLOG_minloglevel=0
# export MAGNUM_LOG=verbose
# export HABITAT_SIM_LOG=verbose

data_dir=$1
sub_split_path=$2
output_dir=$3
filter_existing_path=$4

bc_dataset_path="$data_dir/demos/objectnav/objectnav_hm3d/objectnav_hm3d_hd/{split}/{split}.json.gz"

python run.py \
    --run-type gen \
    --exp-config configs/experiments/il_objectnav.yaml \
    TASK_CONFIG.DATASET.DATA_PATH $bc_dataset_path \
    NUM_ENVIRONMENTS 8 \
    TASK_CONFIG.DATASET.SUB_SPLIT_INDEX_PATH $sub_split_path \
    TASK_CONFIG.REPRESENTATION_GENERATOR.data_storage.output_path $output_dir \
    TASK_CONFIG.REPRESENTATION_GENERATOR.skip_look_actions False \
    TASK_CONFIG.REPRESENTATION_GENERATOR.generate_skip_index False \
    # TASK_CONFIG.DATASET.FILTER_EXISTING_PATH $filter_existing_path \
    # TASK_CONFIG.SIMULATOR.ACTION_SPACE_CONFIG "v1_no_op_look" \
