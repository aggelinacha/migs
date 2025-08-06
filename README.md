# MIGS: Multi-Identity Gaussian Splatting via Tensor Decomposition
## ECCV 2024 (oral)

## [Paper](https://arxiv.org/abs/2407.07284) | [Project Page](https://aggelinacha.github.io/MIGS/)


<img src="./teaser.gif" width="800"/> 


## Installation

Clone this repo:
```shell
git clone --recursive https://github.com/aggelinacha/migs.git
```

Create a conda environment and install dependencies.

This repo has been tested with Python 3.7.13 on Ubuntu 24.04.2 and CUDA 12.8, using the dependencies as exported in `environment.yml`.

```shell
conda env create -f environment.yml
conda activate migs
# install tinycudann
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Our work is built upon [3DGS-Avatar](https://github.com/mikeqzy/3dgs-avatar-release/), we encourage users to check their repo too.

### SMPL Setup
Create a new directory of your choice `/path/to/datasets/smpl`.

Follow the standard SMPL setup (also used in [3DGS-Avatar](https://github.com/mikeqzy/3dgs-avatar-release/)).

Download `SMPL v1.0 for Python 2.7` from [SMPL website](https://smpl.is.tue.mpg.de/) (for male and female models), and `SMPLIFY_CODE_V2.ZIP` from [SMPLify website](https://smplify.is.tue.mpg.de/) (for the neutral model). After downloading, inside `SMPL_python_v.1.0.0.zip`, male and female models are `smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl` and `smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl`, respectively. Inside `mpips_smplify_public_v2.zip`, the neutral model is `smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`. Copy the .pkl models under `/path/to/datasets/smpl/models/*.pkl`.

Remove the chumpy objects in these .pkl models using [this code](https://github.com/vchoutas/smplx/tree/master/tools) under a Python 2 environment (you can create such an environment with conda). Copy newly generated .pkl files under `/path/to/datasets/smpl/models_nochumpy/*.pkl`.

Then, run the following script to extract necessary SMPL parameters used in our code:
```
python extract_smpl_parameters.py
```
The extracted SMPL parameters will be saved into `/path/to/datasets/smpl/misc/`.



## Data

Create a new directory of your choice `/path/to/datasets/zju_mocap_arah`.

### AIST++ dataset

We mostly use the [AIST++](https://google.github.io/aistplusplus_dataset/) dataset in our work.

We have uploaded pre-processed data from the AIST++ dataset [here](https://drive.google.com/drive/folders/13hcscF1p4loORDB4Odram29Sdum-WP7y?usp=drive_link). 
Download them and put them under `/path/to/datasets/zju_mocap_arah/`.

[TODO] We will provide the pre-processing script as well.


### ZJU-Mocap dataset

Please follow the instructions of [ARAH](https://github.com/taconite/arah-release) to download and preprocess ZJU-Mocap.


## Paths

Create a new directory of your choice `/path/to/migs/exp/`, where the results of the experiments will be saved.

Change accordingly the paths under `configs/config.yaml`:
```
exp_dir: /path/to/migs/exp/${name}
smpl_data_path: /path/to/datasets/smpl
```

Change accordingly the path under `configs/dataset/aist_crop_multi.yaml`:
```
root_dir: "/path/to/datasets/zju_mocap_arah"
```

## Training

To train from scratch:
```shell
# AIST++
python train_multi.py dataset=aist_crop_multi model.gaussian.R=100 texture=siren pose_correction=none model.texture.identity_dim=64 model.deformer.non_rigid.identity_dim=64 opt.iterations=100000
# ZJU-MoCap
python train_multi.py dataset=zjumocap_multi model.gaussian.R=100 texture=siren pose_correction=none model.texture.identity_dim=64 model.deformer.non_rigid.identity_dim=64 opt.iterations=100000
```

To train on different subjects, change the subject list (dataset.subjects) in the config file (e.g., `configs/dataset/aist_crop_multi.yaml`).

[UPDATE]: We have improved the output visual quality of the original MIGS architecture by replacing the color MLP with a SIREN MLP. Thus, we use `texture=siren model.texture.identity_dim=64 model.deformer.non_rigid.identity_dim=64` in the training command. 


## Personalization
Fine-tuning to a specific subject for better visual quality and identity-specific details:
```shell
test_subject=gLO_sFM_c01_d15_mLO4_ch19_crop1080;
python train_multi.py dataset=aist_crop_multi dataset_name=aist_crop1080_multi_CP_R100_siren_iter100_finetune_${test_subject} dataset.test_subject=${test_subject} start_checkpoint=/path/to/migs/exp/aist_crop1080_multi_CP_R100_siren_iter100-none-mlp_field-ingp-siren-default/ckpt100000.pth model.gaussian.R=100 texture=siren pose_correction=none model.texture.identity_dim=64 model.deformer.non_rigid.identity_dim=64 opt.iterations=20000 opt.finetune=true  
```


## Test
```shell
test_subject=gLO_sFM_c01_d15_mLO4_ch19_crop1080;
python render_multi.py mode=test dataset=aist_crop_multi dataset.test_subject=${test_subject} texture=siren pose_correction=none model.gaussian.R=100 dataset_name=aist_crop1080_multi_CP_R100_siren_iter100_finetune_${test_subject} load_ckpt=/path/to/migs/exp/aist_crop1080_multi_CP_R100_siren_iter100_finetune_${test_subject}-none-mlp_field-ingp-siren-default/ckpt20000.pth model.texture.identity_dim=64 model.deformer.non_rigid.identity_dim=64
```

Results are saved under `/path/to/migs/exp/aist_crop1080_multi_CP_R100_siren_iter100_finetune_${test_subject}-none-mlp_field-ingp-siren-default/test-view-${test_subject}/`. Metrics are saved in `results.npz`. Rendered frames are saved under `renders/`.


## Animate under novel poses
```shell
test_subject=gLO_sFM_c01_d15_mLO4_ch19_crop1080;
target_poses=gBR_sBM_c01_d04_mBR0_ch01_crop1080;
python render_multi.py mode=predict dataset=aist_crop_multi dataset.test_subject=${test_subject} dataset.predict_seq=${target_poses} texture=siren pose_correction=none model.gaussian.R=100 dataset_name=aist_crop1080_multi_CP_R100_siren_iter100 load_ckpt=/nfs/130.245.4.102/add_disk0/aggelina/results/3dgs/exp/aist_crop1080_multi_CP_R100_siren_iter100_finetune_${test_subject}-none-mlp_field-ingp-siren-default/ckpt20000.pth model.texture.identity_dim=64 model.deformer.non_rigid.identity_dim=64
```

Results are saved under `/path/to/migs/exp/aist_crop1080_multi_CP_R100_siren_iter100_finetune_${test_subject}-none-mlp_field-ingp-siren-default/predict-${test_subject}-${target_poses}/`.


## Acknowledgements
We would like to sincerely thank the authors of the following papers:
- [3DGS-Avatar](https://github.com/mikeqzy/3dgs-avatar-release/)
- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
- [4D-Humans](https://github.com/shubham-goel/4D-Humans)
- [AIST++](https://google.github.io/aistplusplus_dataset/)



## Citation
If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{chatziagapi2024migs,
    title={MIGS: Multi-Identity Gaussian Splatting via Tensor Decomposition},
    author={Aggelina Chatziagapi and Grigorios G. Chrysos and Dimitris Samaras},
    year={2024},
    booktitle={ECCV},
}
```
                
                    