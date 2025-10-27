# [Latent Diffusion Models with Masked AutoEncoders (LDMAE)](https://arxiv.org/pdf/2507.09984)

<!-- ![LDMAE Generation Samples](figure/thumbnail.png)    -->
[Junho Lee](mailto:joon2003@snu.ac.kr)\*, [Jeongwoo Shin](mailto:swswss@snu.ac.kr)\*, [Hyungwook Choi](mailto:chooi221@snu.ac.kr), [Joonseok Lee](mailto:joonseok@snu.ac.kr)†


Seoul National University, Seoul, Korea
\* Equal contribution
† Corresponding author

![ICCV 2025](https://img.shields.io/badge/ICCV-2025-blue)

<p align="center">
  <img src="figure/thumbnail.png" alt="LDMAE Generation Samples" width="70%">
</p>

## Abstract

This project implements **Latent Diffusion Models with Masked AutoEncoders (LDMAE)**, presented at ICCV 2025. We analyze the role of autoencoders in LDMs and identify three key properties: latent smoothness, perceptual compression quality, and reconstruction quality. We demonstrate that existing autoencoders fail to simultaneously satisfy all three properties, and propose Variational Masked AutoEncoders (VMAEs), taking advantage of the hierarchical features maintained by Masked AutoEncoders. Through comprehensive experiments, we demonstrate significantly enhanced image generation quality and computational efficiency.

The codebase is built upon [MAE](https://github.com/facebookresearch/mae) and [LightningDiT](https://github.com/hustvl/LightningDiT).

## Requirements

### Environment Setup

1. Create a conda environment:
```bash
conda create -n ldmae python=3.10
conda activate ldmae
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training Pipeline

### Step 1: Train Autoencoder

First, train the autoencoder model using the VMAE module:

```bash
cd VMAE
bash train_ae.sh
```

The training script includes:
- Autoencoder training
- Positional embedding replacement
- Decoder fine-tuning

After training is complete, save the trained model checkpoint as `vmaef8d16.pth` in the `LDMAE/pretrain_weight/` directory.

** Pretrained checkpoints are also available [HERE](https://drive.google.com/drive/folders/1Cj5Ina4C65C952myawIWUgCwtizu2CXh?usp=drive_link) **

### Step 2: Configure Datasets

Before proceeding with feature extraction and training, configure the dataset paths in the config files located in the `LDMAE/configs/` directory:

- For ImageNet: Edit `configs/imagenet/lightningdit_b_vmae_f8d16_cfg.yaml`
- For CelebA-HQ: Edit `configs/celeba_hq/lightningdit_b_vmae_f8d16_cfg.yaml`

Update the dataset paths according to your local setup.

### Step 3: Feature Extraction

Extract features from your datasets using the trained autoencoder:

#### ImageNet
```bash
cd LDMAE
bash run_extract_feature.sh configs/imagenet/lightningdit_b_vmae_f8d16_cfg.yaml
```

#### CelebA-HQ
```bash
cd LDMAE
bash run_extract_feature.sh configs/celeba_hq/lightningdit_b_vmae_f8d16_cfg.yaml
```

### Step 4: Train Diffusion Model

Train the diffusion model on the extracted features:

#### ImageNet
```bash
bash run_train.sh configs/imagenet/lightningdit_b_vmae_f8d16_cfg.yaml
```

#### CelebA-HQ
```bash
bash run_train.sh configs/celeba_hq/lightningdit_b_vmae_f8d16_cfg.yaml
```

### Step 5: Inference

Generate images using the trained model:

```bash
bash run_inference.sh {CONFIG_PATH}
```

Replace `{CONFIG_PATH}` with the path to your configuration file (e.g., `configs/imagenet/lightningdit_b_vmae_f8d16_cfg.yaml`).

## Project Structure

```
ldmae_for_github/
├── LDMAE/                  # Main diffusion model implementation (based on LightningDiT)
│   ├── configs/           # Configuration files for different datasets
│   ├── datasets/          # Dataset loaders and utilities
│   ├── models/            # Model architectures
│   ├── tokenizer/         # Tokenization modules
│   └── pretrain_weight/   # Directory for pretrained weights
├── VMAE/                   # Masked Autoencoder implementation
│   ├── train_ae.sh        # Autoencoder training script
│   └── ...
└── requirements.txt        # Python dependencies
```

## Configuration Files

The project includes various configuration files for different model variants and datasets:

- **ImageNet configs**: Located in `LDMAE/configs/imagenet/`
- **CelebA-HQ configs**: Located in `LDMAE/configs/celeba_hq/`

Each configuration file specifies:
- Model architecture parameters
- Training hyperparameters
- Dataset paths
- Autoencoder settings

## Notes

- Ensure all dataset paths are correctly configured before training
- The autoencoder must be trained first before feature extraction
- Feature extraction is required before training the diffusion model
- The codebase is based on [LightningDiT](https://github.com/hustvl/LightningDiT) with minimal modifications

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@InProceedings{Lee_2025_ICCV,
    author    = {Lee, Junho and Shin, Jeongwoo and Choi, Hyungwook and Lee, Joonseok},
    title     = {Latent Diffusion Models with Masked AutoEncoders},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {17422-17431}
}
```
```bibtex
@article{lee2025latent,
  title={Latent Diffusion Models with Masked AutoEncoders},
  author={Lee, Junho and Shin, Jeongwoo and Choi, Hyungwook and Lee, Joonseok},
  journal={arXiv preprint arXiv:2507.09984},
  year={2025}
}
```
