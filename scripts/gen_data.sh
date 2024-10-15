export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

bc_dataset_path="../data/habitat/demos/data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_hd/{split}/{split}.json.gz"

CUDA_VISIBLE_DEVICES=1 python run.py \
    --run-type gen \
    --exp-config configs/experiments/il_objectnav.yaml \
    TASK_CONFIG.DATASET.DATA_PATH $bc_dataset_path \
    NUM_ENVIRONMENTS 32 \
    TASK_CONFIG.DATASET.SORT_BY_SCENE_AND_GOAL True \
    TASK_CONFIG.DATASET.EPISODE_STRIDE 10 \
    TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator_name clip \
    TASK_CONFIG.REPRESENTATION_GENERATOR.output_zarr_path /storage/dc/pvr_data/ten_percent/clip_data \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator_name non_visual \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.output_zarr_path /storage/dc/pvr_data/ten_percent/non_visual_data \

