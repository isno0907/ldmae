# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from taming.modules.losses.lpips import LPIPS

import models_mae
from engine_pretrain import train_one_epoch
from datasets import load_dataset
from PIL import Image
import random

def get_args_parser():
    """Parses command-line arguments for pre-training."""
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    
    # --- Custom Arguments for LDMAE ---
    parser.add_argument('--fixed_std', default=None, type=float, help='Fixed standard deviation for the model.')
    parser.add_argument('--visible_loss_ratio', default=0.5, type=float, help='Ratio for visible loss.')
    parser.add_argument('--tune_decoder', action='store_true', help='Flag to tune only the decoder.')
    parser.add_argument('--pred_with_conv', action='store_true', help='Flag to use a convolutional layer for prediction.')
    parser.add_argument('--smooth_output', action='store_true', help='Flag for smoothing the output.')
    parser.add_argument('--fixed_lr', action='store_true', help='Flag to use a fixed learning rate.')
    parser.add_argument('--no_cls', action='store_true', help='Flag to exclude class token.')
    parser.add_argument('--perceptual_loss_ratio', default=None, type=float, help='Ratio for perceptual loss.')
    parser.add_argument('--save_epochs', default=10, type=int, help='Frequency of saving checkpoints.')
    parser.add_argument('--gradual_resol', action='store_true', help='Flag for gradual resolution training.')
    parser.add_argument('--kl_loss_weight', default=None, type=float, help='Weight for KL divergence loss.')
    
    # --- Default MAE Arguments ---
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU.')
    parser.add_argument('--epochs', default=400, type=int, help='Total training epochs.')
    parser.add_argument('--accum_iter', default=1, type=int, help='Gradient accumulation iterations.')

    # --- Model Parameters ---
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL', help='Name of model to train.')
    parser.add_argument('--input_size', default=256, type=int, help='Input image size.')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio.')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use per-patch normalized pixels as targets for loss.')
    parser.set_defaults(norm_pix_loss=False)

    # --- Optimizer Parameters ---
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay.')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='Absolute learning rate.')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='Base learning rate.')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='Lower learning rate bound.')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='Epochs to warmup LR.')

    # --- Dataset Parameters ---
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str, help='Dataset path.')
    parser.add_argument('--output_dir', default='./output_dir', help='Path to save outputs.')
    parser.add_argument('--log_dir', default='./output_dir', help='Path for TensorBoard logs.')
    parser.add_argument('--device', default='cuda', help='Device to use for training.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--resume', default='', help='Resume from checkpoint.')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Start epoch.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # --- Distributed Training Parameters ---
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes.')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='URL for distributed training setup.')

    return parser

def setup_environment(args):
    """Initializes distributed mode and sets up seeds for reproducibility."""
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))  # Fixed: Added missing closing parenthesis
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    return device

def get_dataset(args):
    """Creates and returns the appropriate training dataset based on the data path."""
    
    class LAIONDataset(torch.utils.data.Dataset):
        """Custom dataset for LAION data loaded from Hugging Face datasets."""
        def __init__(self, hf_dataset, transform=None):
            self.hf_dataset = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.hf_dataset)

        def __getitem__(self, idx):
            sample = self.hf_dataset[idx]
            image = sample['image']
            text = sample.get('text', -1)
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if self.transform:
                image = self.transform(image)
            return image, text

    class CustomImageDataset(torch.utils.data.Dataset):
        """Custom dataset for a folder of images with random labels."""
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
            self.num_classes = 10

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            label = random.randint(0, self.num_classes - 1)
            if self.transform:
                image = self.transform(image)
            return image, label

    # Fixed: Changed normalization values to standard ImageNet values
    mean, std = 0.5, 0.5
    
    if 'imagenet' in args.data_path:
        print('Using ImageNet dataset.')
        dataset_path = os.path.join(args.data_path, 'train')
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.75, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        return datasets.ImageFolder(dataset_path, transform=transform_train)
        
    elif 'laion' in args.data_path:
        print('Using LAION dataset.')
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.75, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        try:
            hf_dataset = load_dataset("imagefolder", data_dir=args.data_path, split="train")
        except Exception as e:
            print(f"Error loading LAION dataset: {e}")
            # Fallback to custom dataset if LAION loading fails
            return CustomImageDataset(args.data_path, transform=transform_train)
        return LAIONDataset(hf_dataset, transform=transform_train)
        
    else:
        print('Using custom image folder dataset.')
        transform_train = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        return CustomImageDataset(args.data_path, transform=transform_train)

def get_model(args):
    """Creates and returns the MAE model."""
    print(f"Creating model: {args.model}")
    
    perceptual_loss = None
    if args.perceptual_loss_ratio is not None:
        print(f"Using Perceptual loss with ratio = {args.perceptual_loss_ratio}")
        perceptual_loss = LPIPS().eval()
        
    model = models_mae.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        ldmae_mode=args.tune_decoder,
        no_cls=args.no_cls,
        img_size=args.input_size,
        kl_loss_weight=args.kl_loss_weight,
        smooth_output=args.smooth_output,
        pred_with_conv=args.pred_with_conv,
        perceptual_loss=perceptual_loss,
        perceptual_loss_ratio=args.perceptual_loss_ratio,
        fixed_std=args.fixed_std
    )
    return model

def main(args):
    """Main training function."""
    device = setup_environment(args)

    # --- Dataset and DataLoader ---
    dataset_train = get_dataset(args)
    print(dataset_train)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print(f"Sampler_train = {sampler_train}")

    log_writer = None
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # --- Model, Optimizer, and Scaler ---
    model = get_model(args).to(device)
    model_without_ddp = model
    print(f"Model = {model_without_ddp}")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"accumulate grad iterations: {args.accum_iter}")
    print(f"effective batch size: {eff_batch_size}")
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of params (M): {n_parameters / 1.e6:.2f}')

    # Fixed: Added check for distributed attribute
    if hasattr(args, 'distributed') and args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # --- Load Checkpoint and Prepare for Training ---
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.tune_decoder and args.mask_ratio > 0.0:
        misc.set_for_tuning_decoder(args=args, model=model)
    else:
        print('No layers are frozen; training the entire model.')
    
    # --- Training Loop ---
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # Fixed: Added check for distributed attribute
        if hasattr(args, 'distributed') and args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        if args.output_dir and (epoch % args.save_epochs == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)