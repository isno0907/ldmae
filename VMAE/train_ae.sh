#!/bin/bash

# ===============================================================================
# This script trains an Autoencoder (AE) model in three stages, 
# based on the methods from the paper.
#
# Stage 1: Pre-train the VMAE model on 128x128 images.
# Stage 2: Reset the positional encoding of the pre-trained model to handle 256x256 images.
# Stage 3: Fine-tune the decoder of the model on 256x256 images.
# ===============================================================================


# ===============================================================================
# Stage 1: VMAE Pre-training (128x128)
# ===============================================================================
# Pre-trains the Vision Mae (VMAE) model using distributed training.
#
# Key Arguments:
# --model: Specifies the model architecture 'mae_for_ldmae_f8d16_prev'.
# --input_size: Sets the input image resolution to 128x128.
# --mask_ratio: Sets the ratio of patches to be masked to 0.25.
# --epochs: The model is trained for 400 epochs.
# --output_dir: Saves checkpoints and logs to './work_dir/vmae_before_decoder_finetuning'.
# ===============================================================================
echo "Starting Stage 1: VMAE Pre-training..."
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 1 --node_rank 0 \
    main_pretrain.py \
    --batch_size 128 \
    --no_cls \
    --accum_iter 2 \
    --num_workers 12 \
    --smooth_output \
    --perceptual_loss_ratio 0.5 \
    --fixed_std 1e-3 \
    --model mae_for_ldmae_f8d16_prev \
    --input_size 128 \
    --mask_ratio 0.25 \
    --visible_loss_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 10 \
    --blr 1.0e-4 \
    --weight_decay 0.05 \
    --kl_loss_weight 1e-6 \
    --data_path /data/dataset/imagenet/1K_dataset \
    --output_dir ./work_dir/vmae_before_decoder_finetuning \
    --log_dir ./work_dir/vmae_before_decoder_finetuning

echo "Stage 1 finished."
echo "------------------------------------------------"


# ===============================================================================
# Stage 2: Positional Encoding (PE) Reset
# ===============================================================================
# Resets the positional encoding of a saved checkpoint to adapt the model from
# 128x128 to 256x256 resolution. This script targets the checkpoint from epoch 90.
#
# Key Arguments:
# --model_name: Specifies the model architecture 'mae_for_ldmae_f8d16_prev'.
# --ckpt_dir: Path to the checkpoint file to be modified.
# ===============================================================================
echo "Starting Stage 2: Positional Encoding Reset..."
python pe_reset.py \
    --model_name mae_for_ldmae_f8d16_prev \
    --ckpt_dir ./work_dir/vmae_before_decoder_finetuning/checkpoint-90.pth

echo "Stage 2 finished."
echo "------------------------------------------------"

# ===============================================================================
# Stage 3: Decoder Tuning (256x256)
# ===============================================================================
# Fine-tunes the decoder part of the VMAE model on a higher resolution.
# The encoder weights are frozen.
#
# Key Arguments:
# --tune_decoder: Flag to enable decoder-only tuning.
# --input_size: The input image resolution is increased to 256x256.
# --mask_ratio: Set to 0.0 as we are not masking inputs for this stage.
# --resume: Loads the weights from the pre-trained and PE-reset checkpoint.
# --output_dir: Saves the final model checkpoints and logs to './work_dir/vmae'.
# ===============================================================================
echo "Starting Stage 3: Decoder Tuning..."
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 1 --node_rank 0 \
    main_pretrain.py \
    --no_cls \
    --tune_decoder \
    --perceptual_loss_ratio 10.0 \
    --batch_size 16 \
    --accum_iter 16 \
    --smooth_output \
    --num_workers 12 \
    --model mae_for_ldmae_f8d16_prev \
    --input_size 256 \
    --mask_ratio 0.0 \
    --visible_loss_ratio 0.5 \
    --epochs 10 \
    --save_epochs 1 \
    --warmup_epochs 0 \
    --blr 1.0e-5 \
    --weight_decay 0.05 \
    --kl_loss_weight 0.0 \
    --data_path /data/dataset/imagenet/1K_dataset \
    --output_dir ./work_dir/vmae \
    --log_dir ./work_dir/vmae \
    --resume ./work_dir/vmae_before_decoder_finetuning/checkpoint-90.pth

echo "Stage 3 finished. AE training complete."
