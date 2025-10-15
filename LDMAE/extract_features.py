import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os, yaml
from safetensors.torch import save_file
from datetime import datetime
from datasets.img_latent_dataset import ImgLatentDataset
from tokenizer import models_mae
from tokenizer.sdvae import Diffusers_AutoencoderKL

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def main(args, train_config):
    """
    Run a tokenizer on full dataset and save the features.
    """
    assert torch.cuda.is_available(), "Extract features currently requires at least one GPU."

    # Setup DDP:
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = args.seed + rank
        if rank == 0:
            print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    except:
        print("Failed to initialize DDP. Running in local mode.")
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    model_name = train_config['vae']['model_name'].split("_")[0]
    output_path = os.path.dirname(train_config['data']['origin_path'])
    dataset_name = train_config['data']['name']
    
    # Setup feature folders:
    output_dir = os.path.join(output_path, f'{model_name}_feature_{dataset_name}_{args.data_split}_{args.image_size}')
    if 'sample' in train_config['data']:
        output_dir += '_sample'
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    print(model_name)
    # Create model:

    if model_name == 'vmae':
        arch = 'mae_for_ldmae_f8d16_prev'
        # chkpt = 'pretrain_weight/mae60_kl_f8d16_200ep.pth'
        chkpt = train_config['vae']['weight_path']
        tokenizer = getattr(models_mae, arch)(ldmae_mode=True, no_cls=True, kl_loss_weight=True, smooth_output=True, img_size=args.image_size)
        checkpoint = torch.load(chkpt, map_location='cpu')
        tokenizer = tokenizer.to(device).eval()
        msg = tokenizer.load_state_dict(checkpoint['model'], strict=False)
        if rank == 0:
            print(model_name, msg)
    elif model_name in ['ae','dae', 'vae','sdv3']:
        tokenizer = Diffusers_AutoencoderKL(
                img_size=args.image_size,
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
        # chkpt_dir = "./pretrain_weight/sdv3f8d16.pth"
        chkpt = train_config['vae']['weight_path']
        checkpoint = torch.load(chkpt, map_location='cpu')
        msg = tokenizer.load_state_dict(checkpoint['model'], strict=False)
        if rank == 0:
            print(model_name, msg)
    else:
        raise("")
    

    print(f"{device} GPU - Model loaded")
    # Setup data:
    data_path = train_config['data']['origin_path']
    datasets = [
        ImageFolder(os.path.join(data_path, args.data_split), transform=tokenizer.img_transform(p_hflip=0.0, img_size=args.image_size)),
        ImageFolder(os.path.join(data_path, args.data_split), transform=tokenizer.img_transform(p_hflip=1.0, img_size=args.image_size))
    ]
    samplers = [
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.seed
        ) for dataset in datasets
    ] # Maybe gray scale files are dropped. Need to be fixed.
    loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        ) for dataset, sampler in zip(datasets, samplers)
    ]
    total_data_in_loop = len(loaders[0].dataset)
    if rank == 0:
        print(f"Total data in one loop: {total_data_in_loop}")

    run_images = 0
    saved_files = 0
    latents = []
    latents_flip = []
    labels = []
    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0]
        if run_images % 100 == 0 and rank == 0:
            print(f'{datetime.now()} processing {run_images} of {total_data_in_loop} images')
        
        for loader_idx, data in enumerate(batch_data):
            x = data[0].to(device)
            y = data[1]  # (N,)
            with torch.no_grad():
                if 'sample' in train_config['data']:
                    z = tokenizer._encode(x)
                else:
                    z = tokenizer.encode(x).latent_dist.mode().detach().cpu()  # (N, C, H, W)

            if batch_idx == 0 and rank == 0:
                print('latent shape', z.shape, 'dtype', z.dtype)
            
            if loader_idx == 0:
                latents.append(z)
                labels.append(y)
            else:
                latents_flip.append(z)

        if len(latents) == 10000 // args.batch_size:
            latents = torch.cat(latents, dim=0)
            latents_flip = torch.cat(latents_flip, dim=0)
            labels = torch.cat(labels, dim=0)
            save_dict = {
                'latents': latents,
                'latents_flip': latents_flip,
                'labels': labels
            }
            for key in save_dict:
                if rank == 0:
                    print(key, save_dict[key].shape)
            save_dict = {key: tensor.contiguous().cpu() for key, tensor in save_dict.items()}
            save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
            save_file(
                save_dict,
                save_filename,
                metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
            )
            if rank == 0:
                print(f'Saved {save_filename}')
            
            latents = []
            latents_flip = []
            labels = []
            saved_files += 1

    # save remainder latents that are fewer than 10000 images
    if len(latents) > 0:
        latents = torch.cat(latents, dim=0)
        latents_flip = torch.cat(latents_flip, dim=0)
        labels = torch.cat(labels, dim=0)
        save_dict = {
            'latents': latents,
            'latents_flip': latents_flip,
            'labels': labels
        }
        for key in save_dict:
            if rank == 0:
                print(key, save_dict[key].shape)
        
        save_dict = {key: tensor.contiguous().cpu() for key, tensor in save_dict.items()}
        save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
        save_file(
            save_dict,
            save_filename,
            metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
        )
        if rank == 0:
            print(f'Saved {save_filename}')

    # Calculate latents stats
    dist.barrier()
    if rank == 0:
        dataset = ImgLatentDataset(output_dir, latent_norm=True, sample=train_config['data']['sample'] if 'sample' in train_config['data'] else False,)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", type=str, default='/path/to/your/data')
    parser.add_argument("--data_split", type=str, default='train')
    parser.add_argument("--output_path", type=str, default="/data/dataset/imagenet/")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument('--config', type=str, default='configs/debug.yaml')
    args = parser.parse_args()

    train_config = load_config(args.config)
    main(args, train_config)
