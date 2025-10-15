# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch import inf
import numpy as np
from torchvision.transforms import functional as F

from typing import Optional, Tuple, Union, List

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            sum_dim = self.mean.dim()
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=list(range(1,sum_dim)),
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=list(range(1,sum_dim)),
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean
    
def set_for_tuning_decoder(args, model):
    args.mask_ratio = 0.0
    model.mask_token = None
    for name, param in model.named_parameters():
        if 'decoder' not in name and 'to_latent' not in name:
            param.requires_grad = False
    
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            print(f"{name}: requires_grad = {param.requires_grad}")
    
def set_for_tuning_decoder_vae(args, model):
    for name, param in model.named_parameters():
        if 'post_quant_conv' in name or 'decoder' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # HERE
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True
        
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    from datetime import timedelta
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank,
                                            timeout=timedelta(minutes=30) )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        # self._scaler = torch.cuda.amp.GradScaler()
        self._scaler = torch.amp.GradScaler("cuda")

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        # loss.backward()
        # optimizer.step()
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model_vqvae(args, epoch, model, model_without_ddp, optimizer_ae, optimizer_disc):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer_ae': optimizer_ae.state_dict(),
            'optimizer_disc': optimizer_disc.state_dict(),
            'epoch': epoch,
            'args': args,
        }

        save_on_master(to_save, checkpoint_path)

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)

        
def resize_pos_embed(pos_embed, new_size):
    _, HW, D = pos_embed.shape
    H = int(HW ** 0.5)
    assert H * H == HW
    pos_embed_2d_resized = torch.nn.functional.interpolate(
        pos_embed.reshape(1,H,H,D).permute(0, 3, 1, 2),  # (batch, channels, height, width)
        size=(new_size, new_size),
        mode='bilinear',
        align_corners=False
    ).permute(0, 2, 3, 1).reshape(1,-1,D)  # (batch, height, width, channels)
    
    return pos_embed_2d_resized

def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            
        state_dict = checkpoint['model']
        
        if state_dict['pos_embed'].shape[1] != model_without_ddp.pos_embed.shape[1]:
            new_size = int(model_without_ddp.pos_embed.shape[1] ** 0.5)
            print(f'latent resolution is {new_size} x {new_size}, reshape pos embedding')
            print(f"prev pos embedding size: {state_dict['pos_embed'].shape}")
            state_dict['pos_embed'] = resize_pos_embed(state_dict['pos_embed'], new_size)
            print(f"new pos embedding size: {state_dict['pos_embed'].shape}")
            
            print(f"prev dec pos embedding size: {state_dict['decoder_pos_embed'].shape}")
            state_dict['decoder_pos_embed'] = resize_pos_embed(state_dict['decoder_pos_embed'], new_size)
            print(f"new dec pos embedding size: {state_dict['decoder_pos_embed'].shape}")
        
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print(msg)
        print("Resume checkpoint %s" % args.resume)
        if not args.tune_decoder:
            if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def all_reduce_sum(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        return x_reduce.item()
    else:
        return x

def write_stat(t, num_rows, path, len_dataset):
    t_np = t.numpy().reshape(num_rows, -1)
    if os.path.isfile(path):
        stat = np.loadtxt(path, delimiter=',').astype(np.int64)
        stat = stat.reshape(num_rows, -1)
        stat = np.concatenate((stat, t_np), axis=1)
        save = np.savetxt(path, stat, delimiter=',', fmt='%d')
    else:
        save = np.savetxt(path, t_np, delimiter=',', fmt='%d')
    
    check = np.loadtxt(path, delimiter=',').astype(np.int64)/len_dataset
    print(f'count_convergence is activated.\n{check.round(3)*100}')


import math
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """
 
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas
 
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)
 
    def __len__(self):
        return self.num_samples

@torch.no_grad()
def update_mask(model, data_loader, device, dataset_train, target_attn, 
                mask_ratio = 0.75, ref_cluster = 'eigen', store_mask = False):
    print("Starts upadating informed mask...")
    len_ds = len(dataset_train)
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Upadating informed mask...'
    print_freq = 20
    masks_weights =[]
    mask_indices = []

    for data_iter_step, (samples, _, index, _, path_first, path_second, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        path_first = path_first.to(device, non_blocking=True)
        path_second = path_second.to(device, non_blocking=True)
        index = index.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            if ref_cluster == 'alternate':
                new_ids_shuffle_1, ref_cluster_size = model.forward_encoder_inference(samples, target_attn, 
                                                        mask_ratio = mask_ratio, ref_cluster = 'small',
                                                        return_score = True)
                new_ids_shuffle_2, ref_cluster_size = model.forward_encoder_inference(samples, target_attn, 
                                                        mask_ratio = mask_ratio, ref_cluster = 'small',
                                                        return_score = True, force_flip=True)
                new_ids_shuffle = torch.stack([new_ids_shuffle_1, new_ids_shuffle_2], dim=0) # 2 B N
                new_ids_shuffle = new_ids_shuffle.permute(1,0,2) # B 2 N
                # print(f'alternate: {new_ids_shuffle.shape}')
            else:
                new_ids_shuffle, ref_cluster_size = model.forward_encoder_inference(samples, target_attn, 
                                                        mask_ratio = mask_ratio, ref_cluster = ref_cluster,
                                                        return_score = True)
        # print(torch.cat([index, path_first, path_second]))
        # mask_info = torch.cat([new_ids_shuffle, ref_cluster_size.unsqueeze(-1), path_first.unsqueeze(-1), path_second.unsqueeze(-1)], dim=-1)
        mask_info = new_ids_shuffle
        # mask_index = torch.cat([index.unsqueeze(-1),path_first.unsqueeze(-1), path_second.unsqueeze(-1)], dim=-1)
        if data_iter_step % 200 == 0:
            print(f'ids_shuffle: {new_ids_shuffle.shape}, ref_cluster_size: {ref_cluster_size.shape}')
            print(f'mask_info: {mask_info.shape}')
        if data_iter_step ==0:
            print('Saving...')
            examples = mask_info.detach().cpu().numpy() 
            store_path = f'/data2/projects/jeongwoo/jeongwoo/mae/analysis/convergence/mask_samples_{ref_cluster}'
            save = np.save(store_path, examples)
        masks_weights.append(mask_info)
        # mask_indices.append(mask_index)
    

    masks_weights = torch.cat(masks_weights, dim=0)
    # mask_indices = torch.cat(mask_indices, dim=0)
    print(f'masks_weights: {masks_weights.shape}')
    dist.barrier()
    gather_masks = [torch.ones_like(masks_weights) for _ in range(dist.get_world_size())]
    # gather_mask_index = [torch.ones_like(mask_indices) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_masks, masks_weights)
    # dist.all_gather(gather_mask_index, mask_indices)
    all_mask_weights = torch.cat(gather_masks)
    # all_mask_indices = torch.cat(gather_mask_index)
    
    all_mask_weights = all_mask_weights[:len_ds]
    # all_mask_indices = all_mask_indices[:len_ds]
    
    if store_mask:
        weights_to_store = all_mask_weights.cpu().numpy()
        store_path = f'/data2/projects/jeongwoo/jeongwoo/mae/analysis/convergence/stored_masks_{ref_cluster}'
        save = np.save(store_path, weights_to_store)
    
    dataset_train.mask = all_mask_weights.cpu()
    # dataset_train.mask_index = all_mask_indices.cpu()
    print("Informed masks have been updated")


import torchvision.transforms as transforms
class maskRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, **kwargs):
        super().__init__(size, **kwargs)
        self.mask_size = 14
    
    def forward(self, img, mask):
        mask = mask.reshape(14,14)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        m_h_s = int(14 * (i/img.size[1]))
        m_h_e = int(14 * ((i+h)/img.size[1])) + 1
        m_w_s = int(14 * (j/img.size[0]))
        m_w_e = int(14 * ((j+w)/img.size[0])) + 1
        mask = mask[m_h_s:m_h_e, m_w_s:m_w_e]

        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        mask = F.resize(mask.unsqueeze(0), (self.mask_size, self.mask_size))
        mask = mask.flatten()
        
        return img, mask
    
class maskRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self):
        super().__init__()

    def forward(self, img, mask):
        mask = mask.reshape(14, 14)
        if torch.rand(1) < self.p:
            img, mask = F.hflip(img), F.hflip(mask)
        mask = mask.flatten()
        return img, mask

class trainCompose(transforms.Compose):
    def __call__(self, img, mask, hint_prob=False): 

        #mask needs to be processed individually for some operations
        for i in self.transforms[:2]:
            img, mask = i(img,mask)

        # if not hint_prob:
        #     mask = torch.argsort(
        #         mask, dim=0, descending=False
        #     ) 
        
        for t in self.transforms[2:]:
            img = t(img)

        return img, mask
    
# def schedule_hint(hint_ratio, hint_portion, do_schedule, cur_epoch, total_epoch):
#     if hint_ratio is None: return None
#     L = 196
#     if do_schedule:
#         alpha = 1 - (cur_epoch/total_epoch)**3 # 1 to 0
#         hint_ratio = hint_ratio * alpha
#         hint_portion = max(hint_portion, 0.2)
#         hint_portion = alpha * (hint_portion - 0.2) + 0.2
#     cluster_size = int(hint_portion*L)
#     hint_num = max(int(hint_ratio * cluster_size), 2)
#     print(f'{hint_num} tokens for hint in epoch {cur_epoch}')
#     return hint_num
def schedule_hint(hint_ratio, hint_portion, do_schedule, cur_epoch, total_epoch, min_portion, min_ratio, schedule_exp, full_schedule = False):
    if hint_ratio is None: return None
    L = 196
    if do_schedule:
        assert hint_portion >= min_portion, 'min_portion is bigger than hint_portion.'
        assert hint_ratio >= min_ratio, 'min_ratio is bigger than hint_ratio.'
        
        if full_schedule:
            total_epoch = 800
            alpha = 1 - ((cur_epoch-0)/(total_epoch-0))**schedule_exp # 1 to 0
        else:
            alpha = 1 - ((cur_epoch-100)/(total_epoch-100))**schedule_exp # 1 to 0
        
        
        hint_ratio = alpha * (hint_ratio - min_ratio) + min_ratio
        hint_portion = alpha * (hint_portion - min_portion) + min_portion
        
    hint_num = max(int(hint_ratio * L), 2)
    print(f'{hint_num} tokens for hint in epoch {cur_epoch}')
    print(f'Hint ratio & hint_portion: {hint_ratio, hint_portion} in epoch {cur_epoch}')
    return hint_ratio, hint_portion


import torchvision.datasets as datasets
import random
class NormalImgDataset(datasets.ImageFolder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_retries = 10
        
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        failed = []
        for _ in range(self.num_retries):
            path, target = self.samples[index]
            try:
                sample = self.loader(path)
            except:
                try:
                    sample = self.loader(path) # one more time
                except:
                    failed.append(path)
                    index = random.randint(0, len(self.samples) - 1)
                    continue
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            return sample, target, torch.tensor(1)
        else:
            print('Failed to load {} after {} retries'.format(
                failed, self.num_retries
            ))