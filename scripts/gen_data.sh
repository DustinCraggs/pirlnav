export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

data_dir=$1

bc_dataset_path="$1/demos/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_hd/{split}/{split}.json.gz"

CUDA_VISIBLE_DEVICES=1 python run.py \
    --run-type gen \
    --exp-config configs/experiments/il_objectnav.yaml \
    TASK_CONFIG.DATASET.DATA_PATH $bc_dataset_path \
    NUM_ENVIRONMENTS 40 \
    TASK_CONFIG.REPRESENTATION_GENERATOR.batch_chunk_size 1000 \
    TASK_CONFIG.DATASET.SUB_SPLIT_INDEX_PATH data/pvr_demos/one_percent/ep_index.json \
    TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.name non_visual \
    TASK_CONFIG.REPRESENTATION_GENERATOR.output_zarr_path data/pvr_demos/one_percent/non_visual_data \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.name clip \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.output_zarr_path data/pvr_demos/ten_percent/clip \

