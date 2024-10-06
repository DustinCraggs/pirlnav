export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

bc_dataset_path="../data/habitat/demos/data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_hd/{split}/{split}.json.gz"

CUDA_VISIBLE_DEVICES=1 python run.py \
    --run-type gen \
    --exp-config configs/experiments/il_objectnav.yaml \
    TASK_CONFIG.DATASET.DATA_PATH $bc_dataset_path \
    TASK_CONFIG.DATASET.EPISODE_STRIDE 200 \
    TASK_CONFIG.DATASET.SORT_BY_SCENE_AND_GOAL True
