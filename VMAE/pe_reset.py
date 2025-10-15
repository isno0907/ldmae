#!/usr/bin/env python
# reset_pe.py
"""
Adjust positional embeddings in an LDMAE checkpoint to match the current model
resolution, then save the updated checkpoint as `<original>_pe.pth`.
"""

import argparse
import os
import sys
import torch

# Add project root (where `models_mae` is located) to the Python path
sys.path.append("..")
import models_mae
# Make sure the import path for `resize_pos_embed` matches your project layout
from models_mae.util.pos_embed import resize_pos_embed  


def reset_positional_embedding(
    chkpt_path: str,
    arch: str = "mae_for_ldmae_f8d16_prev",
    *,
    input_size: int = 256,
    ldmae_mode: bool = True,
    no_cls: bool = True,
    gradual_resol: bool = False,
    smooth_output: bool = True,
    pred_with_conv: bool = False,
    kl_loss_weight=None,
    use_initialized_pe: bool = False,
    modify_dec_pred: bool = False,
) -> str:
    """
    Resize the positional embeddings in `chkpt_path` to fit the target model.

    Returns
    -------
    str
        Path to the newly saved checkpoint.
    """
    ckpt_all = torch.load(chkpt_path, map_location="cpu")
    ckpt = ckpt_all["model"]

    # Create a dummy model to obtain the target PE shape
    model = getattr(models_mae, arch)(
        ldmae_mode=ldmae_mode,
        no_cls=no_cls,
        img_size=input_size,
        gradual_resol=gradual_resol,
        smooth_output=smooth_output,
        pred_with_conv=pred_with_conv,
        kl_loss_weight=kl_loss_weight,
    )

    # Resize PE if the resolution has changed
    if model.pos_embed.shape[1] != ckpt["pos_embed"].shape[1]:
        new_size = int(model.pos_embed.shape[1] ** 0.5)
        ckpt["pos_embed"] = resize_pos_embed(ckpt["pos_embed"], new_size)
        ckpt["decoder_pos_embed"] = resize_pos_embed(ckpt["decoder_pos_embed"], new_size)

    # Optionally convert decoder predictor keys to a 2-layer MLP format
    if modify_dec_pred:
        ckpt["decoder_pred.linear_pred.bias"] = ckpt.pop("decoder_pred.bias")
        ckpt["decoder_pred.linear_pred.weight"] = ckpt.pop("decoder_pred.weight")

    # Optionally remove PE to force random re-initialization
    if use_initialized_pe:
        ckpt.pop("pos_embed", None)
        ckpt.pop("decoder_pos_embed", None)

    ckpt_all["model"] = ckpt
    save_path = os.path.splitext(chkpt_path)[0] + "_pe.pth"
    torch.save(ckpt_all, save_path)
    print(f"[+] Saved adjusted checkpoint → {save_path}")
    return save_path


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Adjust positional embeddings in an LDMAE checkpoint"
    )
    parser.add_argument(
        "--model_name",
        default="mae_for_ldmae_f8d16_prev",
        help="Architecture registered in models_mae",
    )
    parser.add_argument(
        "--chkpt_dir",
        required=True,
        help="Path to the original checkpoint (.pth)",
    )
    # Expose extra flags if needed, e.g. --input_size
    return parser.parse_args()


def main():
    args = parse_args()
    reset_positional_embedding(
        chkpt_path=args.chkpt_dir,
        arch=args.model_name,
    )


if __name__ == "__main__":
    main()
