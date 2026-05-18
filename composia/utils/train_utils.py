import torch
import torch.nn.functional as F
import gc
from pathlib import Path
from typing import List
from decord import VideoReader
from contextlib import contextmanager
import cv2
from PIL import Image
import os
import logging
import torchvision.transforms.functional as TF
from typing import Optional, Tuple, List, Union
import numpy as np
from einops import rearrange

logger = logging.getLogger(__name__)

WAN_FUN_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

##########  loaders  ##########

def load_prompts(prompt_path: Path) -> List[str]:
    with open(prompt_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_videos(video_path: Path) -> List[Path]:
    with open(video_path, "r", encoding="utf-8") as file:
        return [video_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_images(image_path: Path) -> List[Path]:
    with open(image_path, "r", encoding="utf-8") as file:
        return [image_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0]
    
@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

def extract_first_frame(video_path: str) -> Image.Image:
    """
    Extract the first frame from a video and save as image.
    
    Args:
        video_path (str): Path to the video file 
        
    Returns:
        Image.Image: The first frame of the video, or None if failed
    """
    # Skip if image already exists
    try:
        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            # Get first frame
            first_frame = video_reader[0].asnumpy()  # Shape: (H, W, C)
            pil_image = Image.fromarray(first_frame)
            
            return pil_image

    except Exception as e:
        logger.error(f"Failed to extract first frame from {video_path}: {e}")
        return None
    
### input preprocess utils for training and inference ###

def resize_mask(mask: torch.Tensor, latent: Optional[torch.Tensor], process_first_frame_only: bool = True, latent_size: Optional[tuple] = None) -> torch.Tensor:
    """Resize a binary mask (B, C, T, H, W) to match latent (B, C, T', H', W').

    - If process_first_frame_only, upscale the first frame to T'=1 and the rest to T'-1, then concat.
    - Accept latent=None when latent_size is provided explicitly.
    """
    if latent_size is None:
        if latent is None:
            raise ValueError("Either latent or latent_size must be provided")
        latent_size = latent.size()

    if mask.dim() != 5:
        raise ValueError(f"mask must be 5D (B,C,T,H,W), got {mask.shape}")

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )

        target_size = list(latent_size[2:])
        target_size[0] = max(target_size[0] - 1, 0)
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )
    return resized_mask

def batch_encode_vae(
    tensor: torch.Tensor,
    vae,
    num_cameras: int,
    mini_batch: int =1,
) -> torch.Tensor:
    """Encode (B, N*F, C, H, W) to latents (B, C, N*F, H//s, W//s) with VAE in mini-batches.

    - The input should be laid out as (B, N*F, C, H, W).
    - This matches training code path and avoids OOM by chunking on (B*N).
    """
    pixel_values = rearrange(tensor, "b (n f) c h w -> (b n) c f h w", n=num_cameras)
    new_pixel_values = []
    for i in range(0, pixel_values.shape[0], mini_batch):
        pixel_values_bs = pixel_values[i : i + mini_batch]
        pixel_values_bs = vae.encode(pixel_values_bs)[0]
        pixel_values_bs = pixel_values_bs.sample()
        new_pixel_values.append(pixel_values_bs)
    new_pixel_values = torch.cat(new_pixel_values, dim=0)
    new_pixel_values = rearrange(new_pixel_values, "(b n) c f h w -> b c (n f) h w", n=num_cameras)
    return new_pixel_values



def prepare_clip_context(
    clip_image_encoder,
    clip_pixel_values: Optional[torch.FloatTensor] = None,
) -> Optional[torch.Tensor]:
    """
    Build clip_context for training:
      - clip_pixel_values: iterable of single images (H,W,3) as tensors on CPU
      - Returns concatenated contexts of shape (B, C_ctx, T)
    """
    if clip_pixel_values is not None:
        clip_image = Image.fromarray(np.uint8(clip_pixel_values.float().cpu().numpy()))
        clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(clip_image_encoder.device, clip_image_encoder.dtype)
        clip_context = clip_image_encoder([clip_image[:, None, :, :]])
    else:
        # fallback to black image, then zeros_like
        black = Image.new("RGB", (512, 512), color=(0, 0, 0))
        img_t = TF.to_tensor(black).sub_(0.5).div_(0.5).to(clip_image_encoder.device, clip_image_encoder.dtype)
        ctx = clip_image_encoder([img_t[:, None, :, :]])
        clip_context = torch.zeros_like(ctx)
    
    return clip_context

def _repeat_first_frame_concat_rest(x: torch.Tensor, repeat_times: int = 4) -> torch.Tensor:
    """Given (B, C, F, H, W), make time F' = F+3 by repeating the first frame 4 times then concatenating the rest."""
    return torch.concat(
        [
            torch.repeat_interleave(x[:, :, 0:1], repeats=repeat_times, dim=2),
            x[:, :, 1:],
        ],
        dim=2,
    )


def prepare_mask_condition(
    mask: torch.Tensor,
    latents: torch.Tensor,
    num_cameras: int,
    temporal_compression_ratio: int = 4,
) -> torch.Tensor:
    """Convert raw binary mask (B, N*F, C, H, W) to resized latent mask (B, C, N*F, H', W').

    Steps follow the training/inference logic:
    - pad by repeating first frame 4x, then group every 4 frames
    - resize to match latent temporal/spatial shape
    """
    mask = rearrange(mask, "b (n f) c h w -> (b n) c f h w", n=num_cameras)
    mask = _repeat_first_frame_concat_rest(mask, repeat_times=temporal_compression_ratio).contiguous()
    mask = mask.view(
        mask.shape[0], mask.shape[2] // temporal_compression_ratio, temporal_compression_ratio, mask.shape[3], mask.shape[4]
    )
    mask = mask.transpose(1, 2)  # (B*N, 4, F', H, W)

    b, c, nf, h, w = latents.size()
    n = num_cameras
    mask_condition = resize_mask(1 - mask, latent=None, latent_size=(b * n, c, nf // n, h, w), process_first_frame_only=False)
    mask_condition = rearrange(mask_condition, "(b n) c f h w -> b c (n f) h w", n=num_cameras)
    
    return mask_condition


def prepare_bbox_mask_condition(
    bbox_masks: torch.Tensor,
    latents: torch.Tensor,
    num_cameras: int,
) -> torch.Tensor:
    """Convert raw binary bbox mask (B, N*F, 1, H, W) to resized latent mask (B, 1, N*F, H', W').

    Steps follow the training/inference logic:
    - pad by repeating first frame 4x, then group every 4 frames
    - resize to match latent temporal/spatial shape
    """
    bbox_masks = rearrange(bbox_masks, "b (n f) c h w -> (b n) c f h w", n=num_cameras)
    b, c, nf, h, w = latents.size()
    n = num_cameras
    bbox_mask_conditoin = resize_mask(bbox_masks, latent=None, latent_size=(b * n, c, nf // n, h, w))
    bbox_mask_conditoin = rearrange(bbox_mask_conditoin, "(b n) c f h w -> b c (n f) h w", n=num_cameras)
    return bbox_mask_conditoin


def prepare_plucker_condition(plucker_embs: torch.Tensor, num_cameras: int, temporal_compression_ratio: int=4) -> torch.Tensor:
    """Pack temporal dimension into channel groups of 4 for input shaped (B, N*F, C, H, W).

    Returns (B, C*4, N*F', H, W).
    """
    dt = temporal_compression_ratio
    x = rearrange(plucker_embs, "b (n f) c h w -> (b n) c f h w", n=num_cameras)
    x = _repeat_first_frame_concat_rest(x, repeat_times=dt).transpose(1, 2).contiguous() # b f c h w
    x = x.view(x.shape[0], x.shape[1] // dt, dt, x.shape[2], x.shape[3], x.shape[4])
    x = x.transpose(2, 3).contiguous()
    x = x.view(x.shape[0], x.shape[1], x.shape[2] * dt, x.shape[4], x.shape[5])
    x = x.transpose(1, 2)
    x = rearrange(x, "(b n) c f h w -> b c (n f) h w", n=num_cameras)
    return x


def prepare_action_condition(actions: torch.Tensor, temporal_compression_ratio: int=4) -> torch.Tensor:
    """action F to diffusion F'
    [B, F, 2] -> [B, F', 2]
    """
    bn, f, c = actions.size()
    a = torch.concat(
        [
            torch.repeat_interleave(actions[:, 0:1], repeats=temporal_compression_ratio, dim=1),
            actions[:, 1:],
        ],
        dim=1,
    )
    df = (f + temporal_compression_ratio-1) // temporal_compression_ratio
    a = a.view(bn, df, temporal_compression_ratio, c)
    a = a.mean(dim=2)
    return a


# def sample_proj_mats(proj_mats: torch.Tensor, num_cameras: int) -> torch.Tensor:
#     """Sample projection matrices by keeping the first frame in each group of 4 along time.

#     Input: (B, N*F, H, W). Output: (B, N*F', H, W) where F' = ceil(F/4).
#     """
#     x = rearrange(proj_mats, "b (n f) h w -> (b n) f h w", n=num_cameras)
#     sample_indexs = [0] + [1 + i * 4 for i in range((x.shape[1] - 1) // 4)]
#     x = x[:, sample_indexs, :, :]
#     x = rearrange(x, "(b n) f h w -> b (n f) h w", n=num_cameras)
#     return x


def sample_proj_mats(proj_mats: torch.Tensor, num_cameras: int, temporal_compression_ratio: int =4) -> torch.Tensor:
    """Sample projection matrices by keeping the first frame in each group of 4 along time.

    Input: (B, N*F, 4, 4). Output: (B, N*F', 4, 4) where F' = ceil(F/4).
    """
    try:
        from pytorch3d.transforms import matrix_to_quaternion
    except ImportError as exc:
        raise ImportError(
            "pytorch3d is required only when using proj_mats conditions. "
            "Install a wheel matching your PyTorch/CUDA environment before enabling proj_mats."
        ) from exc

    x = rearrange(proj_mats, "b (n f) h w -> (b n) f h w", n=num_cameras)
    trans_norm = torch.norm(x[:, :, 0:3, 3], dim=-1)  # (B*N, F)
    non_zero_trans = trans_norm > 1e-8
    norm_factor = torch.sum(trans_norm, dim=-1) / torch.sum(non_zero_trans, dim=-1)  # (B*N,)

    x0 = x[:, 0:1, :, :].repeat(1, temporal_compression_ratio, 1, 1)
    if x.shape[1] > 1:
        x_rest = x[:, 1:, :, :]
        x = torch.cat([x0, x_rest], dim=1)
    else:
        x = x0
    quat = matrix_to_quaternion(x[:, :, 0:3, 0:3])  # (B*N, F, 4)
    normalized_trans = x[:, :, 0:3, 3] / norm_factor[:, None, None]  # (B*N, F, 3)
    log_norm_factor = torch.log(norm_factor + 1e-8).unsqueeze(-1).unsqueeze(-1).repeat(1, x.shape[1], 1)  # (B*N, F, 1)

    quat = rearrange(quat, "b (f d) c -> b f (d c)", d=temporal_compression_ratio)  # (B*N, F//4, 16)
    normalized_trans = rearrange(normalized_trans, "b (f d) c -> b f (d c)", d=temporal_compression_ratio)  # (B*N, F//4, 12)
    log_norm_factor = rearrange(log_norm_factor, "b (f d) c -> b f (d c)", d=temporal_compression_ratio)  # (B*N, F//4, 4)


    quat = rearrange(quat, "(b n) f c -> b (n f) c", n=num_cameras)
    normalized_trans = rearrange(normalized_trans, "(b n) f c -> b (n f) c", n=num_cameras)
    log_norm_factor = rearrange(log_norm_factor, "(b n) f c -> b (n f) c", n=num_cameras)

    return torch.cat([quat, normalized_trans, log_norm_factor], dim=-1)  # (B, N*F // 4, 32)

def get_consistent_texture_video(tex_big, target_shape):
    """
    Use one shared crop for the whole video and apply per-frame random flips.
    target_shape: (T, 3, H, W) or (B, T, 3, H, W)
    """
    device, dtype = tex_big.device, tex_big.dtype
    H, W = target_shape[-2], target_shape[-1]
    
    # Pick one random crop location shared by the whole video.
    # `tex_big` has shape [3, H_tex, W_tex].
    H_tex, W_tex = tex_big.shape[-2:]
    t = torch.randint(0, H_tex - H + 1, (1,)).item()
    l = torch.randint(0, W_tex - W + 1, (1,)).item()
    
    # Crop the base patch and expand it across frames.
    base_tex = tex_big[:, t:t+H, l:l+W].unsqueeze(0)
    
    N = torch.Size(target_shape).numel() // (3 * H * W)
    video_bg = base_tex.expand(N, -1, -1, -1).clone() # Clone so each frame can be flipped independently.

    # Apply random flips independently for each frame.
    flip_h = torch.rand(N, 1, 1, 1, device=device) > 0.5
    flip_v = torch.rand(N, 1, 1, 1, device=device) > 0.5
    
    # Apply the sampled flips.
    video_bg = torch.where(flip_h, video_bg.flip(-1), video_bg)
    video_bg = torch.where(flip_v, video_bg.flip(-2), video_bg)

    return video_bg.view(*target_shape)
