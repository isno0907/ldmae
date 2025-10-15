# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Mlp, DropPath
from typing import Dict, Optional, Tuple, Union
import os, sys

sys.path.append("./tokenizer")
from util.pos_embed import get_2d_sincos_pos_embed

############################## Docking Functions ##############################
from dataclasses import dataclass
from typing import Optional, Tuple
from diffusers.utils import BaseOutput
from einops import rearrange
from diffusers import ConfigMixin, ModelMixin
from util.misc import DiagonalGaussianDistribution
from PIL import Image
from torchvision import transforms
import numpy as np
from taming.modules.losses.lpips import LPIPS


class Config:
    def __init__(self, scaling_factor):
        self.scaling_factor = scaling_factor

@dataclass
class DecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.Tensor
    commit_loss: Optional[torch.FloatTensor] = None

@dataclass
class EncoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """
    
    latent: torch.Tensor

    def sample(self):
        return self.latent
    def mode(self):
        return self.latent

@dataclass
class MAEOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: Union[DiagonalGaussianDistribution, EncoderOutput]  # noqa: F821

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

####################################################################################


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attn_map = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if return_attn_map:
            qk_attn = attn.clone().detach()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ctxed = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x_ctxed)
        x = self.proj_drop(x)
        
        if return_attn_map:
            return x, [qk_attn, x_ctxed]
        return x
    
class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, return_attn_map = False):
        if return_attn_map:
            x_tmp, qk_and_x = self.attn(self.norm1(x), return_attn_map = True)
            # returned attn is Q @ K / scale, before softmax
        else:
            x_tmp = self.attn(self.norm1(x))
        x = x + self.drop_path1(self.ls1(x_tmp))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        
        if return_attn_map:
            return x, qk_and_x
        return x

class Downsample(nn.Module): # from downblock in AutoencoderKL
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, 3, stride=2)
        
    def forward(self, x): # x : B N C
        B, N, C = x.shape
        H = int(N ** 0.5)
        assert H * H == N, 'Size mismatch.'
        x = x.reshape(B, H, H, C).permute(0,3,1,2) # B C H W
        
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x) # B C H W
        
        x = x.reshape(B, C, -1).permute(0,2,1) # B N C
        return x
    
class Upsample(nn.Module): # from upblock in AutoencoderKL
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, 3, 1, 1)
        
    def forward(self, x): # x : B N C
        B, N, C = x.shape
        H = int(N ** 0.5)
        assert H * H == N, 'Size mismatch.'
        x = x.reshape(B, H, H, C).permute(0,3,1,2) # B C H W
        
        if x.shape[0] >= 64:
            x = x.contiguous()
            
        scale_factor = 2
        if x.numel() * scale_factor > pow(2, 31):
            x = x.contiguous()

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        
        x = self.conv(x)
        
        x = x.reshape(B, C, -1).permute(0,2,1) # B N C
        return x
    
class MLP_dim_resize(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_dim_resize, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.GELU(),                         
            nn.Linear(hidden_dim, output_dim) 
        )

    def forward(self, x):
        return self.layers(x)
    
class conv_decoder_pred(nn.Module):
    def __init__(self, decoder_embed_dim, patch_size, in_chans, pred_with_conv=True):
        super(conv_decoder_pred, self).__init__()
        self.p = patch_size
        self.pred_with_conv = pred_with_conv
        if self.pred_with_conv:
            print('pred only with conv instead of previous linear')
            self.conv_smoother = nn.Conv2d(decoder_embed_dim, patch_size**2 * in_chans, 2, stride=1, padding=0)
        else:
            print('conv on rgb')
            self.linear_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
            self.conv_smoother = nn.Conv2d(in_chans, in_chans, 3, 1, 1)
            
    def forward(self, x):
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        if self.pred_with_conv:
            B = x.shape[0]
            x = x.reshape(B, h, w, -1).permute(0,3,1,2)
            padding = (0, 1, 0, 1)  # Pad 1 on the right (W) and 1 on the bottom (H)
            # Apply padding
            x = F.pad(x, padding, mode='constant', value=0)
            x  = self.conv_smoother(x) # B C H W
            x = x.reshape(B, -1, h*w).permute(0,2,1) # B HW C
            
        else:
            x = self.linear_pred(x) # B HW p_size*p_size*3
            x = x.reshape(shape=(x.shape[0], h, w, self.p, self.p, 3))
            x = torch.einsum('nhwpqc->nchpwq', x)
            x = x.reshape(shape=(x.shape[0], 3, h * self.p, w * self.p)) # B 3 256 256
            
            x = self.conv_smoother(x)
            x = x.reshape(x.shape[0], 3, h, self.p, w, self.p)
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(x.shape[0], h*w, self.p*self.p*3)) # B HW C
        
        return x

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 latent_dim=32, ldmae_mode=False, scaling_factor=0.9654248952865601, no_cls=True, 
                 gradual_resol=False, finetune_downsample_layer=None, down_nonlinear=False,
                 kl_loss_weight=None, smooth_output=False, pred_with_conv=False, perceptual_loss=None):
        super().__init__()
        
        # --------------------------------------------------------------------------
        # MAE for LDMAE settings
        self.perceptual_loss = perceptual_loss
        self.smooth_output = smooth_output
        self.gradual_resol = gradual_resol
        self.kl_loss_weight = kl_loss_weight
        encoder_latent_dim = latent_dim
        decoder_latent_dim = latent_dim
        if self.kl_loss_weight is not None: 
            assert no_cls, 'There should be no class token to use KL loss.'
            encoder_latent_dim = 2 * latent_dim
            print(f'Use KL loss, encoder latent dim is {encoder_latent_dim} to predict mean & logvar')
        if self.gradual_resol:
            patch_size = patch_size // 2
            print(f'patch size: {patch_size}')
        
        if down_nonlinear:
            print('Use MLP for latent embedding')
            self.to_latent = MLP_dim_resize(embed_dim, latent_dim*4, encoder_latent_dim)
            self.from_latent = MLP_dim_resize(decoder_latent_dim, latent_dim*4, embed_dim)
        else:
            self.to_latent = nn.Linear(embed_dim, encoder_latent_dim)
            self.from_latent = nn.Linear(decoder_latent_dim, decoder_embed_dim)
        self.config = Config(scaling_factor=scaling_factor)
        self.ldmae_mode = ldmae_mode
        self.img_size = img_size
        self.patch_size = patch_size
        self.latent_resolution = img_size // patch_size
        self.tile_latent_min_size = self.latent_resolution
        self.latent_dim = latent_dim
        self.no_cls = no_cls
        self.num_extra_tokens = 0
        # if head_output_num is not None:
        #     self.head = torch.nn.ModuleList([
        #         torch.nn.LayerNorm(self.latent_dim),
        #         # torch.nn.BatchNorm1d(self.latent_dim, affine=False, eps=1e-6),
        #         torch.nn.Linear(in_features=self.latent_dim, out_features=head_output_num, bias=False)
        #     ])
            # self.head_norm = torch.nn.LayerNorm(self.latent_dim)
            # self.head_bn = torch.nn.BatchNorm1d(self.latent_dim, affine=False, eps=1e-6)
            # self.head = torch.nn.Linear(in_features=self.latent_dim, out_features=head_output_num)
        if not self.no_cls:
            self.num_extra_tokens += 1
        # --------------------------------------------------------------------------
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        if not self.no_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_extra_tokens, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        if self.gradual_resol:
            blocks = []
            downsize_time = depth // 2 if finetune_downsample_layer is None else finetune_downsample_layer
            for i in range(depth):
                blocks.append(Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
                if i == downsize_time-1:
                    print(f"Add downsizing block in {i}th layer in encoder.")
                    blocks.append(Downsample(embed_dim, embed_dim))
            self.blocks = nn.ModuleList(blocks)
        else:
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        if not self.ldmae_mode:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if self.gradual_resol:
            decoder_num_patches = num_patches // 4
            assert decoder_num_patches * 4 == num_patches
        else:
            decoder_num_patches = num_patches
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, decoder_num_patches + self.num_extra_tokens, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        if self.gradual_resol:
            decoder_blocks = []
            upsize_time = decoder_depth - downsize_time
            for i in range(decoder_depth):
                decoder_blocks.append(Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
                if i == upsize_time-1:
                    print(f"Add upsizing block in {i}th layer in decoder.")
                    decoder_blocks.append(Upsample(decoder_embed_dim, decoder_embed_dim))
            self.decoder_blocks = nn.ModuleList(decoder_blocks)
        else:
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])
            
        self.decoder_norm = norm_layer(decoder_embed_dim)
        if smooth_output:
            assert no_cls, 'Should be no CLS token for smooth_output.'
            print('Use conv in decoder pred.')
            self.decoder_pred = conv_decoder_pred(decoder_embed_dim, patch_size, in_chans, pred_with_conv)
        else:
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
    
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token= not self.no_cls)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.gradual_resol:
            decoder_num_patches = self.patch_embed.num_patches // 4
        else:
            decoder_num_patches = self.patch_embed.num_patches
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(decoder_num_patches**.5), cls_token= not self.no_cls)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if not self.no_cls:
            torch.nn.init.normal_(self.cls_token, std=.02)
        if not self.ldmae_mode:
            torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        
        # add pos embed w/o cls token
        if self.no_cls:
            x = x + self.pos_embed
        else:
            x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        if not self.no_cls:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        if not self.no_cls:
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        else:
            x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        if not self.no_cls:
            x = x[:, 1:, :]

        return x
    
    def forward_encoder_with_mask(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        
        # add pos embed w/o cls token
        # if self.no_cls:
        #     x = x + self.pos_embed
        # else:
        #     x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # if not self.no_cls:
        #     x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        #     x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        #     x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # else:
        x_ = torch.cat([x, mask_tokens], dim=1) 
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # append cls token
        if self.no_cls:
            x = x + self.pos_embed
        else:
            x = x + self.pos_embed[:, 1:, :]
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder_without_mask(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # if not self.no_cls:
        #     x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        #     x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        #     x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # else:
        #     x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        #     x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        if not self.no_cls:
            x = x[:, 1:, :]

        return x
    
    # self.no_cls에 따라 다르게 encoding & decoding
    # decoding은 이후에 unpatchfy 넣어주어야함
    def ldmae_encoding(self, imgs, use_mode=False, return_kl=False):
        # encoding
        x = self.patch_embed(imgs)
        if not self.no_cls:
            x = x + self.pos_embed[:, 1:, :]
        else:
            x = x + self.pos_embed
        if not self.no_cls:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # to latent
        latent = self.to_latent(x)

        if self.kl_loss_weight is not None:
            latent = latent.permute(0,2,1) # B D HW
            posterior = DiagonalGaussianDistribution(latent) # B D HW
            if use_mode:
                latent = posterior.mode() # B D HW
            else:
                latent = posterior.sample() # B D HW
            latent = latent.permute(0,2,1)
        if return_kl:
            return latent, posterior.kl()
        return latent
    
    def ldmae_decoding(self, x):
        # if self.smooth_output:
        #     with torch.no_grad():
        #         # from latent
        #         x = self.from_latent(x)
                
        #         x = self.decoder_embed(x)
        #         # if x.shape[1] != self.decoder_pos_embed.shape[1]:
        #         if not self.no_cls:
        #             decoder_pos_embed = self.decoder_pos_embed[:,1:,:]
        #         else:
        #             decoder_pos_embed = self.decoder_pos_embed
        #         x = x + decoder_pos_embed
        #         for blk in self.decoder_blocks:
        #             x = blk(x)
        #         x = self.decoder_norm(x)
        # else:
        # from latent
        x = self.from_latent(x)
        
        x = self.decoder_embed(x)
        # if x.shape[1] != self.decoder_pos_embed.shape[1]:
        if not self.no_cls:
            decoder_pos_embed = self.decoder_pos_embed[:,1:,:]
        else:
            decoder_pos_embed = self.decoder_pos_embed
        x = x + decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
            
        x = self.decoder_pred(x)
        if not self.no_cls:
            x = x[:, 1:, :]
        return x
    
    def reconstruct(self, imgs, use_mode=True, return_kl=False):
        with torch.no_grad():
            if return_kl:
                x, kl_val = self.ldmae_encoding(imgs, use_mode=use_mode, return_kl=return_kl)
            else:
                x = self.ldmae_encoding(imgs, use_mode=use_mode, return_kl=return_kl)
        x = self.ldmae_decoding(x)
        if return_kl:
            return x, kl_val
        return x
    
    def linear_probe_seg(self, images):
        with torch.no_grad():
            x = self.ldmae_encoding(images)

        if not self.no_cls:
            x = x[:, 1:, :] # B HW D
        
        B, N, D = x.shape
        x = x.reshape(-1, D) # BHW D
        
        for layer in self.head:
            x = layer(x)
        # x = self.head(x)
        return x # BHW num_classes
    
    def linear_probe(self, images):
        with torch.no_grad():
            x = self.ldmae_encoding(images)
            
        # x = self.head_norm(x)
        if self.no_cls:
            x = x.mean(dim=1)  # global pool, [B D]
        else:
            x = x[:, 1:, :].mean(dim=1)  # global pool, [B D]
        # x = self.head_bn(x)
        for layer in self.head:
            x = layer(x)
        return x
    
    def forward_loss(self, imgs, pred, mask, visible_loss_ratio=0.5):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        visible_loss = (loss * (1-mask)).sum() / (1-mask).sum()
        mask_loss = (loss * mask).sum() / mask.sum()
        
        loss = (1-visible_loss_ratio) * mask_loss + visible_loss_ratio * visible_loss
        
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss, visible_loss, mask_loss
    
    def forward_vanilla(self, imgs, mask_ratio=0.75, visible_loss_ratio=0.5):
        if self.gradual_resol:
            latent, mask, ids_restore = self.forward_encoder_with_mask(imgs, mask_ratio)
        else:
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
            
        # --------------------------------------------------------------------------
        # to latent
        latent = self.to_latent(latent)

        if self.kl_loss_weight is not None:
            B, N, D = latent.shape
            latent = latent.permute(0,2,1) # B D HW
            posterior = DiagonalGaussianDistribution(latent) # B D HW
            
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0] / N # per patch
            
            latent = posterior.sample() # B D HW
            latent = latent.permute(0,2,1)
            
        latent = self.from_latent(latent)
        # --------------------------------------------------------------------------
        
        if self.gradual_resol:
            pred = self.forward_decoder_without_mask(latent, ids_restore)  # [N, L, p*p*3]
        else:
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            
        loss, vis_loss, mask_loss = self.forward_loss(imgs, pred, mask, visible_loss_ratio)
        if self.kl_loss_weight is not None:
            loss = loss + self.kl_loss_weight * kl_loss
        else:
            kl_loss = None
        return loss, pred, mask, vis_loss, mask_loss, kl_loss
    
    def forward_ldmae(self, imgs):
        pred = self.reconstruct(imgs, use_mode=False)
        
        # target = self.patchify(imgs)
        vis_loss = (self.unpatchify(pred) - imgs) ** 2
        
        if self.perceptual_loss is not None:
            p_loss = self.perceptual_loss(
                imgs.contiguous(),
                self.unpatchify(pred).contiguous()
            )
            loss = vis_loss + p_loss
        else:
            loss = vis_loss
            p_loss = vis_loss
        
        loss = loss.mean()
        return loss, pred, None, vis_loss.mean(), p_loss.mean(), None
    
    def forward(self, imgs, mask_ratio=0.75, visible_loss_ratio=0.5):
        if self.ldmae_mode:
            return self.forward_ldmae(imgs)
        else:
            return self.forward_vanilla(imgs, mask_ratio=mask_ratio, visible_loss_ratio=visible_loss_ratio)
    
    ############################ Docking Functions ####################################
    
    def _encode(self, x):
        # encoding
        x = self.patch_embed(x)
        if not self.no_cls:
            x = x + self.pos_embed[:, 1:, :]
        else:
            x = x + self.pos_embed
        if not self.no_cls:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # to latent
        x = self.to_latent(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.latent_resolution, w=self.latent_resolution)
        return x
    
    def encode(self, x, return_dict=True):
        # encoding
        x = self.patch_embed(x)
        if not self.no_cls:
            x = x + self.pos_embed[:, 1:, :]
        else:
            x = x + self.pos_embed
        if not self.no_cls:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # to latent
        x = self.to_latent(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.latent_resolution, w=self.latent_resolution)

        if self.kl_loss_weight is not None:
            p = DiagonalGaussianDistribution(x) # B c h w
        else:
            p = EncoderOutput(x)
        if not return_dict:
            return (p,)
        
        return MAEOutput(latent_dist=p)

    def decode(self, z, return_dict=True, generator=None):
        
        # from latent
        z = rearrange(z, 'b c h w -> b (h w) c')
        x = self.from_latent(z)
        
        x = self.decoder_embed(x)
        # if x.shape[1] != self.decoder_pos_embed.shape[1]:
        if not self.no_cls:
            decoder_pos_embed = self.decoder_pos_embed[:,1:,:]
        else:
            decoder_pos_embed = self.decoder_pos_embed
        x = x + decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        img = self.unpatchify(x)
        if not return_dict:
            return (img,)
        
        return DecoderOutput(sample=img)

    @property
    def device(self) -> torch.device:
        """
        Returns:
            torch.device: The torch device on which the model's parameters are located.
        """
        # Check if the model has parameters that specify a device
        for param in self.parameters():
            return param.device

        # Check if the model has buffers (like self.pos_embed or self.cls_token)
        for buffer in self.buffers():
            return buffer.device

        # Fallback to CPU if no parameters or buffers are found
        return torch.device("cpu")
    
    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
        return self.device
    
    @property
    def dtype(self) -> torch.dtype:
        """
        Returns:
            torch.dtype: The torch data type on which the model's parameters are stored.
        """
        # Check if the model has any parameters to infer dtype
        for param in self.parameters():
            return param.dtype

        # Check if the model has buffers to infer dtype
        for buffer in self.buffers():
            return buffer.dtype

        # Default to float32 if no parameters or buffers are found
        return torch.float32
    
    #################################################################################################
    
    ####################################### VAVAE Docking Functions #################################
    def img_transform(self, p_hflip=0, img_size=None):
        """Image preprocessing transforms
        Args:
            p_hflip: Probability of horizontal flip
            img_size: Target image size, use default if None
        Returns:
            transforms.Compose: Image transform pipeline
        """
        img_size = img_size if img_size is not None else self.img_size
        img_transforms = [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
            transforms.RandomHorizontalFlip(p=p_hflip),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ]
        return transforms.Compose(img_transforms)
    
    def encode_images(self, images):
        """Encode images to latent representations
        Args:
            images: Input image tensor
        Returns:
            torch.Tensor: Encoded latent representation
        """
        with torch.no_grad():
            posterior = self.encode(images.cuda(), return_dict=False)[0]
            return posterior.sample()

    def decode_to_images(self, z):
        """Decode latent representations to images
        Args:
            z: Latent representation tensor
        Returns:
            np.ndarray: Decoded image array
        """
        with torch.no_grad():
            images = self.decode(z.cuda(), return_dict=False)[0]
            images = torch.clamp(127.5 * images + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        return images
    
    ############################################################################################

def mae_for_ldmae(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=128, patch_size=8, embed_dim=192, depth=12, num_heads=12,
        decoder_embed_dim=192, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), latent_dim=32, **kwargs)
    return model

# 위랑 같은건데 그냥 복붙해서 이름만 바꿈
def mae_for_ldmae_f8d32(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=128, patch_size=8, embed_dim=192, depth=12, num_heads=12,
        decoder_embed_dim=192, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), latent_dim=32, **kwargs)
    return model

def mae_for_ldmae_f8d16_prev(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8, embed_dim=192, depth=12, num_heads=12,
        decoder_embed_dim=192, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), latent_dim=16, **kwargs)
    return model

def mae_for_ldmae_f8d16_prev_large(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8, embed_dim=384, depth=12, num_heads=16,
        decoder_embed_dim=384, decoder_depth=12, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), latent_dim=16, **kwargs)
    return model

def mae_for_ldmae_f8d16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8, embed_dim=192, depth=12, num_heads=12,
        decoder_embed_dim=384, decoder_depth=12, decoder_num_heads=24,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), latent_dim=16, down_nonlinear=True,**kwargs)
    return model

def mae_for_ldmae_f8d16_flexible(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8, embed_dim=192, depth=12, num_heads=12,
        decoder_embed_dim=384, decoder_depth=12, decoder_num_heads=24,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), latent_dim=16, down_nonlinear=True,**kwargs)
    return model

def mae_for_ldmae_f16d32(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=128, patch_size=16, embed_dim=192, depth=12, num_heads=12,
        decoder_embed_dim=192, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), latent_dim=32, **kwargs)
    return model

def mae_for_ldmae_f16d32_large(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=128, patch_size=16, embed_dim=384, depth=12, num_heads=12,
        decoder_embed_dim=384, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), latent_dim=32, finetune_downsample_layer=4, **kwargs)
    return model

def mae_for_ldmae_f8d32_flexible(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8, embed_dim=192, depth=12, num_heads=12,
        decoder_embed_dim=192, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), latent_dim=32, **kwargs)
    return model

def mae_for_ldmae_16d(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=128, patch_size=8, embed_dim=192, depth=12, num_heads=12,
        decoder_embed_dim=192, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), latent_dim=16, **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec128d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch16_128 = mae_vit_base_patch16_dec128d8b
