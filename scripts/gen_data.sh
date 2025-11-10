export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

data_dir=$1
sub_split_path=$2
output_dir=$3
clip_model_path=None
# clip_model_path="/data/drive2/models/clip-vit-base-patch32"
# cogvlm2_model_path="/storage/dc/models/vlm/cogvlm2-llama3-chat-19B-int4/"

bc_dataset_path="$data_dir/demos/objectnav/objectnav_hm3d/objectnav_hm3d_hd/{split}/{split}.json.gz"

python run.py \
    --run-type gen \
    --exp-config configs/experiments/il_objectnav.yaml \
    TASK_CONFIG.DATASET.DATA_PATH $bc_dataset_path \
    NUM_ENVIRONMENTS 20 \
    TASK_CONFIG.DATASET.SUB_SPLIT_INDEX_PATH $sub_split_path \
    TASK_CONFIG.REPRESENTATION_GENERATOR.data_storage.output_path $output_dir \
    TASK_CONFIG.REPRESENTATION_GENERATOR.skip_look_actions False \
    TASK_CONFIG.REPRESENTATION_GENERATOR.generate_skip_index False \
    # TASK_CONFIG.SIMULATOR.ACTION_SPACE_CONFIG "v1_no_op_look" \
    # TASK_CONFIG.DATASET.FILTER_EXISTING_PATH /home/dc/data/nav_datasets/pirlnav_costmap_datasets/ten_percent/gt_perception_graphs_split_1/ep_index.txt \
    # TASK_CONFIG.DATASET.FILTER_EXISTING_PATH /home/dc/data/nav_datasets/pirlnav_costmap_datasets/ten_percent/test/ep_index.txt \
    # DATASET.CONTENT_SCENES 
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.clip_kwargs.model_path $clip_model_path \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.cogvlm2_kwargs.model_path $cogvlm2_model_path \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.name cogvlm2 \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.output_zarr_path $output_dir/cogvlm2 \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.name non_visual \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.output_zarr_path $output_dir/non_visual_data \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.name clip \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.output_zarr_path $output_dir/clip_data \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.name vc_1 \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.output_zarr_path $output_dir/vc_1_data \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.data_generator.name raw_image \
    # TASK_CONFIG.REPRESENTATION_GENERATOR.output_zarr_path $output_dir/raw_image_data \