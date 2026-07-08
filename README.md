# New

## Installation

```bash
git submodule update --init

conda create -n pirlnav python=3.10 cmake=3.14.0

conda activate pirlnav

cd habitat-sim/
pip install -r requirements.txt
python setup.py install --headless

cd ..
pip install -r habitat-lab/habitat_baselines/il/requirements.txt
pip install -e habitat-lab
pip install -e .

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install zarr ifcfg einops strictfire natsort hydra-core wandb numpy==1.26 opencv-python==4.10.0.84 tensorboard

# Apply my patch to habitat-lab:
git -C habitat-lab apply ../habitat_lab.patch
```

## Data Generation

Can symlink datasets to `data/`:
- `data/demos`
- `data/scene_datasets` (`versioned_data/hm3d-1.0/`)
- `data/tasks`
- `data/visual_encoders`

Generate the data:

```bash
./scripts/gen_data.sh \
    data \
    data/zarr/ten_percent/split_0/ten_eps_ep_index.json \
    data/zarr/ten_percent/split_0/test_rgb
```

- This creates a zarr group at path `data/zarr/ten_percent/split_0/test_rgb` containing
  RGB and non-visual observations (can be configured in
  `configs/tasks/objectnav_hm3d.yaml` under `REPRESENTATION_GENERATOR`)
- There's only one scene in the test split above, so there will be many duplicate eps

## Train:

NOTE: Edit the following according to your setup:
- `NUM_ENVIRONMENTS` and `IL.BehaviorCloning` configs in `scripts/pvr_il.sh`
- The wandb config in `configs/experiments/il_objectnav.yaml`
- Make sure to change the run name (`test_run`) between different runs, otherwise it
  will try to resume

```bash
./scripts/pvr_il.sh \
    data \
    data/zarr/ten_percent/split_0/test_rgb/ \
    data/zarr/ten_percent/split_0/test_rgb/ \
    None \
    0 \
    test_run_encoder_thaw_2 \
    test_group \
    1 \
    1.0 \
    3
```

## Eval:

Evaluate on the train episodes as a simple test:

```bash
./scripts/eval_train.sh \
    data \
    data/zarr/ten_percent/split_0/test_rgb/ \
    data/zarr/ten_percent/split_0/test_rgb/ \
    data/checkpoints/objectnav_il/test_run_4/ckpt.9.pth \
    eval_train_test_run \
    eval_train_test_group \
    data/zarr/ten_percent/split_0/ten_eps_ep_index.json
```

Evaluate on the validation episodes (held-out scenes):

```bash
./scripts/eval.sh \
    data \
    data/zarr/ten_percent/split_0/test_rgb/ \
    data/zarr/ten_percent/split_0/test_rgb/ \
    data/checkpoints/objectnav_il/test_run_4/ckpt.9.pth \
    eval_test_run \
    eval_test_group
```

NOTE: Edit `NUM_ENVIRONMENTS` for more parallelism based on available resources
when evaluating on the full dataset.

## Generate a new split (e.g. a bigger one):

Example with stride 20, starting idx 0:

```
scripts/gen_sub_split.sh data data/zarr/stride_20_ep_index.json 20 0
```

Note: This samples diverse scene-goal pairs by sorting by (scene, goal) and then
sampling at the desired stride. The best way to do it would be to pool eps into a list
for each scene-goal pair and then round-robin sample until the desired ep count is
reached. If necessary, I can implement this.

# PIRLNav: Pretraining with Imitation and RL Finetuning for ObjectNav

Code for our paper [PIRLNav: Pretraining with Imitation and RL Finetuning for ObjectNav](https://arxiv.org/pdf/2301.07302.pdf). 

Ram Ramrakhya, Dhruv Batra, Erik Wijmans, Abhishek Das

[Project Page](https://ram81.github.io/projects/pirlnav)


## What is PIRLNav?

PIRLNav is a two-stage learning scheme for IL pretraining on human demonstrations followed by RL-finetuning for ObjectNav. To enable successful RL finetuning, we present a two-stage learning scheme involving a critic-only learning phase first that gradually transitions over to training both the actor and critic. 

<p align="center">
  <img src="imgs/teaser.png"  height="400">

  <p align="center"><i>Scaling laws of <code>IL→RL</code> for ObjectNav </i></p>
</p>

Using this IL→RL training recipe, we present a rigorous empirical analysis of design choices. We study how
RL-finetuning performance scales with the size of the IL pretraining dataset. We find that as we increase the size of the IL-pretraining dataset and get to high IL accuracies, the improvements from RL-finetuning are smaller, and that 90% of the performance of our best IL→RL policy can be achieved with less than half the number of IL demonstrations.

Read more in the [paper]().


## Installation

New:

```bash
git submodule update --init

conda create -n pirlnav python=3.10 cmake=3.14.0

# Uninstall any other versions of habitat_sim or habitat-lab

cd habitat-sim/
pip install -r requirements.txt
python setup.py install --headless

cd ..
pip install -r habitat-lab/habitat_baselines/il/requirements.txt
pip install -e habitat-lab
pip install -e .

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# pip install zarr ifcfg einops strictfire natsort hydra-core wandb==0.19.8 numpy==1.24.0
pip install zarr ifcfg einops strictfire natsort hydra-core wandb numpy==1.26

# For Ray, transformers:
pip install "ray[serve]" transformers

# Phoenix:
module load Mesa/23.1.4-GCCcore-12.3.0
# Maybe:
conda install -c conda-forge libstdcxx-ng sysroot_linux-64
conda install -c conda-forge libgl libegl

# sg_habitat:
pip install transformers h5py kornia
conda install -c conda-forge spatialmath-python
pip install git+https://github.com/openai/CLIP.git
```

Run the following commands:

```
git clone https://github.com/Ram81/pirlnav.git
git submodule update --init

conda create -n pirlnav python=3.7 cmake=3.14.0

cd habitat-sim/
pip install -r requirements.txt
./build.sh --headless

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

cd habitat-lab/
pip install -r requirements.txt

pip install -e habitat-lab
pip install -e habitat-baselines

pip install -e .
```


## Data

### Downloading HM3D Scene and Episode Dataset

- Download the HM3D dataset using the instructions [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d) (download the full HM3D dataset for use with habitat)

- Move the HM3D scene dataset or create a symlink at `data/scene_datasets/hm3d`.

- Download the ObjectNav HM3D episode dataset from [here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md#task-datasets).


### Download Demonstrations Dataset

You can use the following datasets to reproduce results reported in our paper.

| Dataset| Scene dataset | Split | Link | Extract path |
| ----- | --- | --- | --- | --- |
| ObjectNav-HD | HM3D | 77k | [objectnav_hm3d_hd.json.gz](https://habitat-on-web.s3.amazonaws.com/pirlnav_release/objectnav_hm3d_hd.zip) | `data/datasets/objectnav/objectnav_hm3d_hd/` |
| ObjectNav-SP | HM3D | 240k | [objectnav_hm3d_sp.json.gz](https://habitat-on-web.s3.amazonaws.com/pirlnav_release/objectnav_hm3d_sp.zip) | `data/datasets/objectnav/objectnav_hm3d_sp/` |
| ObjectNav-FE | HM3D | 70k | [objectnav_hm3d_fe.json.gz](https://habitat-on-web.s3.amazonaws.com/pirlnav_release/objectnav_hm3d_fe.zip) | `data/datasets/objectnav/objectnav_hm3d_fe/` |

The demonstration datasets released as part of this project are licensed under a [Creative Commons Attribution-NonCommercial 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/legalcode).


### OVRL Encoder Weights

To train policies using OVRL pretrained RGB encoder, download the model weights from [here](https://habitat-on-web.s3.amazonaws.com/pirlnav_release/checkpoints/omnidata_DINO_02.pth) and move to `data/visual_encoders/`.

### Dataset Folder Structure

The code requires the datasets in `data` folder in the following format:

  ```bash
  ├── habitat-web-baselines/
  │  ├── data
  │  │  ├── scene_datasets/
  │  │  │  ├── hm3d/
  │  │  │  │  ├── JeFG25nYj2p.glb
  │  │  │  │  └── JeFG25nYj2p.navmesh
  │  │  ├── datasets
  │  │  │  ├── objectnav/
  │  │  │  │  ├── objectnav_hm3d/
  │  │  │  │  │  ├── objectnav_hm3d_hd/
  │  │  │  │  │  │   ├── train/
  │  │  │  │  │  ├── objectnav_hm3d_v1/
  │  │  │  │  │  │   ├── train/
  │  │  │  │  │  │   ├── val/
  │  │  ├── visual_encoders
  ```

## Usage

### IL Training


For training the behavior cloning policy on the ObjectGoal Navigation task use the following script:

  ```bash
  sbatch scripts/1-objectnav-il.sh <dataset_name>
  ```

  where `dataset_name` can be `objectnav_hm3d_hd`, `objectnav_hm3d_sp`, or `objectnav_hm3d_fe`

### RL Finetuning

For RL finetuning the behavior cloned policy on the ObjectGoal Navigation task use the following script:

  ```bash
  sbatch scripts/2-objectnav-rl-ft.sh /path/to/initial/checkpoint
  ```

### Evaluation

To evaluate a checkpoint trained using behavior cloning checkpoint use the following command:

  ```bash
  sbatch scripts/1-objectnav-il-eval.sh /path/to/checkpoint
  ```

For evaluating a checkpoint trained using RL finetuning use the following command: 

  ```bash
  sbatch scripts/1-objectnav-rl-ft-eval.sh /path/to/checkpoint
  ```


## Reproducing Results

We provide best checkpoints for agents trained on ObjectNav task with imitation learning and RL finetuning. You can use the following checkpoints to reproduce results reported in our paper.

| Task | Checkpoint | Success Rate | SPL |
| --- | --- | --- | --- |
| 🆕[ObjectNav](https://arxiv.org/abs/2006.13171) | [objectnav_il_hd.ckpt](https://habitat-on-web.s3.amazonaws.com/pirlnav_release/checkpoints/objectnav_il_hd.ckpt) | 64.1 | 27.1 |
| 🆕[ObjectNav](https://arxiv.org/abs/2006.13171) | [objectnav_rl_ft_hd.ckpt](https://habitat-on-web.s3.amazonaws.com/pirlnav_release/checkpoints/objectnav_rl_ft_hd.ckpt) | 70.4 | 34.1 |


## Citation

If you use this code in your research, please consider citing:

```
@inproceedings{ramrakhya2023pirlnav,
      title={PIRLNav: Pretraining with Imitation and RL Finetuning for ObjectNav},
      author={Ram Ramrakhya and Dhruv Batra and Erik Wijmans and Abhishek Das},
      booktitle={CVPR},
      year={2023},
}
```

