"""
Evaluate tokenizer performance by computing reconstruction metrics.

Metrics include:
- rFID (Reconstruction FID)
- PSNR (Peak Signal-to-Noise Ratio) 
- LPIPS (Learned Perceptual Image Patch Similarity)
- SSIM (Structural Similarity Index)

by Jingfeng Yao
from HUST-VL
"""

import os
import torch, yaml
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from tools.calculate_fid import calculate_fid_given_paths
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchmetrics import StructuralSimilarityIndexMeasure
from models.lpips import LPIPS
from torchvision.datasets import ImageFolder
from torchvision import transforms
from diffusers.models import AutoencoderKL
from tokenizer.sdvae import Diffusers_AutoencoderKL
from tokenizer import models_mae

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def print_with_prefix(content, prefix='Tokenizer Evaluation', rank=0):
    if rank == 0:
        print(f"\033[34m[{prefix}]\033[0m {content}")

def save_image(image, filename):
    Image.fromarray(image).save(filename)

def evaluate_tokenizer(args, config_path, model_type, data_path, output_path):
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    train_config = load_config(config_path)
    model_type = train_config['vae']['model_name'].split("_")[0]

    if local_rank == 0:
        print_with_prefix(f"Loading model... {model_type.upper()} {args.epsilon}")
    
    if train_config['vae']['model_name'].split("_")[0] == 'vmae':
        chkpt = train_config['vae']['weight_path']
        arch = 'mae_for_ldmae_f8d16_prev'
        model = getattr(models_mae, arch)(ldmae_mode=True, no_cls=True, kl_loss_weight=True, smooth_output=True, img_size=train_config['data']['image_size'])
        checkpoint = torch.load(chkpt, map_location='cpu')
        model = model.to(device).eval()
        msg = model.load_state_dict(checkpoint['model'], strict=False)
    elif train_config['vae']['model_name'].split("_")[0] in ['ae','dae','vae','sdv3']:
        model = Diffusers_AutoencoderKL(
            img_size=train_config['data']['image_size'],
            sample_size=128,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            latent_channels=16,
            norm_num_groups=32,
            act_fn="silu",
            block_out_channels=(128, 256, 512, 512),
            force_upcast=False,
            use_quant_conv=False,
            use_post_quant_conv=False,
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            up_block_types=(
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
        ).to(device).eval()
        chkpt_dir = train_config['vae']['weight_path']
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
    else:
        raise
    print(msg)
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create dataset and dataloader
    dataset = ImageFolder(root=data_path, transform=transform)
    distributed_sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=local_rank)
    val_dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        sampler=distributed_sampler
    )
    
    if 'sample' in train_config['data']:
        train_config['data']['data_path'] += '_sample'
    latent_stats_cache_file = os.path.join(train_config['data']['data_path'], 'latents_stats.pt')
    latent_stats = torch.load(latent_stats_cache_file)
    latent_mean, latent_std = latent_stats['mean'], latent_stats['std']
    
    latent_mean = latent_mean.clone().detach().to(device)
    latent_std = latent_std.clone().detach().to(device)
    

    # Setup output directories
    folder_name = f"{model_type}_{args.epsilon}"
    
    save_dir = os.path.join(output_path, folder_name, 'decoded_images')
    ref_path = os.path.join(output_path, 'ref_images')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ref_path, exist_ok=True)

    if local_rank == 0:
        print_with_prefix(f"Output dir: {save_dir}")
        print_with_prefix(f"Reference dir: {ref_path}")

    # Save reference images if needed
    ref_png_files = [f for f in os.listdir(ref_path) if f.endswith('.png')]
    if len(ref_png_files) < 50000:
        total_samples = 0
        for batch in val_dataloader:
            images = batch[0].to(device)
            for j in range(images.size(0)):
                img = torch.clamp(127.5 * images[j] + 128.0, 0, 255).cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                Image.fromarray(img).save(os.path.join(ref_path, f"ref_image_rank_{local_rank}_{total_samples}.png"))
                total_samples += 1
                if total_samples % 100 == 0 and local_rank == 0:
                    print_with_prefix(f"Rank {local_rank}, Saved {total_samples} reference images")
    dist.barrier()

    # Initialize metrics
    lpips_values = []
    ssim_values = []
    lpips = LPIPS().to(device).eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0)).to(device)

    # Generate reconstructions and compute metrics
    if local_rank == 0:
        print_with_prefix("Generating reconstructions...")
    all_indices = 0
    if len(os.listdir(save_dir)) < 50000:
        for batch in val_dataloader:
            images = batch[0].to(device)
            latents = encode_images(model, images)
            epsilon = args.epsilon * torch.randn_like(latents)
            latents = latents + epsilon * latent_std
            
            with torch.no_grad():
                decoded_images_tensor = model.decode(latents).sample            
                decoded_images = torch.clamp(127.5 * decoded_images_tensor + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            
            # Compute metrics
            lpips_values.append(lpips(decoded_images_tensor, images).mean())
            ssim_values.append(ssim_metric(decoded_images_tensor, images))
            
            # Save reconstructions
            for i, img in enumerate(decoded_images):
                save_image(img, os.path.join(save_dir, f"decoded_image_rank_{local_rank}_{all_indices + i}.png"))
                if (all_indices + i) % 100 == 0 and local_rank == 0:
                    print_with_prefix(f"Rank {local_rank}, Processed {all_indices + i} images")
            all_indices += len(decoded_images)
    dist.barrier()

    # Aggregate metrics across GPUs
    lpips_values = torch.tensor(lpips_values).to(device)
    ssim_values = torch.tensor(ssim_values).to(device)
    dist.all_reduce(lpips_values, op=dist.ReduceOp.AVG)
    dist.all_reduce(ssim_values, op=dist.ReduceOp.AVG)
    
    avg_lpips = lpips_values.mean().item()
    avg_ssim = ssim_values.mean().item()

    if local_rank == 0:
        # Calculate FID
        print_with_prefix("Computing rFID...")
        fid = calculate_fid_given_paths([ref_path, save_dir], batch_size=50, dims=2048, device=device, num_workers=16)

        # Calculate PSNR
        print_with_prefix("Computing PSNR...")
        psnr_values = calculate_psnr_between_folders(ref_path, save_dir)
        avg_psnr = sum(psnr_values) / len(psnr_values)

        # Print final results
        print_with_prefix(f"Final Metrics:")
        print_with_prefix(f"rFID: {fid:.3f}")
        print_with_prefix(f"PSNR: {avg_psnr:.3f}")
        print_with_prefix(f"LPIPS: {avg_lpips:.3f}")
        print_with_prefix(f"SSIM: {avg_ssim:.3f}")
    dist.barrier()
    dist.destroy_process_group()

def encode_images(model, images):
    with torch.no_grad():
        posterior = model.encode(images).latent_dist
        return posterior.mode().to(torch.float32)

def decode_to_images(model, z):
    with torch.no_grad():
        images = model.decode(z)
        images = torch.clamp(127.5 * images + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return images

def calculate_psnr(original, processed):
    mse = torch.mean((original - processed) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse)).item()

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32)

def calculate_psnr_for_pair(original_path, processed_path):
    return calculate_psnr(load_image(original_path), load_image(processed_path))

def calculate_psnr_between_folders(original_folder, processed_folder):
    original_files = sorted(os.listdir(original_folder))
    processed_files = sorted(os.listdir(processed_folder))

    if len(original_files) != len(processed_files):
        print("Warning: Mismatched number of images in folders")
        return []

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(calculate_psnr_for_pair,
                          os.path.join(original_folder, orig),
                          os.path.join(processed_folder, proc))
            for orig, proc in zip(original_files, processed_files)
        ]
        return [future.result() for future in as_completed(futures)]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='tokenizer/configs/vavae_f16d32.yaml')
    parser.add_argument('--model_type', type=str, default='vavae')
    parser.add_argument('--data_path', type=str, default='/data/dataset/imagenet/1K_dataset/val')
    parser.add_argument('--output_path', type=str, default='./rfid')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epsilon', type=float, default=0, help="Noise pertubation ratio for latent robustness experiment.")
    args = parser.parse_args()
    evaluate_tokenizer(args, config_path=args.config_path, model_type=args.model_type, data_path=args.data_path, output_path=args.output_path)