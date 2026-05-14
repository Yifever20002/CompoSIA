# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import glob
import json
import math
import os
import types
import warnings
import functools
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version, logging
from diffusers.models.controlnet import zero_module
from torch import nn
from einops import rearrange

from ..dist import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    xFuserLongContextAttention)
from ..dist.wan_xfuser import usp_attn_forward
from ..utils.logger import logger
from .cache_utils import TeaCache, cfg_skip, disable_cfg_skip, enable_cfg_skip
from torch.nn.attention import flex_attention
from accelerate.logging import get_logger

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    major, minor = torch.cuda.get_device_capability(0)
    if f"{major}.{minor}" == "8.0":
        from sageattention_sm80 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif f"{major}.{minor}" == "8.6":
        from sageattention_sm86 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif f"{major}.{minor}" == "8.9":
        from sageattention_sm89 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif major>=9:
        from sageattention_sm90 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
except:
    try:
        from sageattention import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    except:
        sageattn = None
        SAGE_ATTENTION_AVAILABLE = False

LOG_NAME = "trainer"
LOG_LEVEL = "INFO"
logger = get_logger(LOG_NAME, LOG_LEVEL)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    attn_mask=None,
):
    attention_type = os.environ.get("VIDEOX_ATTENTION_TYPE", "FLASH_ATTENTION")
    if attention_type == "SAGE_ATTENTION" and SAGE_ATTENTION_AVAILABLE:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = sageattn(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
    elif attention_type == "FLASH_ATTENTION" and (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        # attn_mask = None

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
    return out


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def sinusoidal_embedding_batchwise(dim, position):
    if len(position.shape) <= 1:
        return sinusoidal_embedding_1d(dim, position)  # for compatibility

    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    orig_shape = position.shape
    position_flat = position.reshape(-1)

    # calculation
    freq = torch.pow(10000, -torch.arange(half, dtype=torch.float64, device=position.device) / half)
    sinusoid = position_flat.unsqueeze(-1) * freq  # [N, half]
    emb = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=-1)  # [N, dim]
    emb = emb.reshape(*orig_shape, dim)
    
    return emb

@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))

    freqs = torch.polar(torch.ones_like(freqs), freqs)  # shape: (S, D/2)
    return freqs

# modified from https://github.com/thu-ml/RIFLEx/blob/main/riflex_utils.py
@torch.amp.autocast('cuda', enabled=False)
def get_1d_rotary_pos_embed_riflex(
    pos: Union[np.ndarray, int],
    dim: int,
    theta: float = 10000.0,
    use_real=False,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
    L_test_scale: Optional[int] = None,
):
    """
    RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
        L_test (`int`, *optional*, defaults to None): the number of frames for inference
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    freqs = 1.0 / torch.pow(theta,
        torch.arange(0, dim, 2).to(torch.float64).div(dim))

    # === Riflex modification start ===
    # Reduce the intrinsic frequency to stay within a single period after extrapolation (see Eq. (8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply by 0.9 to keep the extrapolated length below 90% of a single period.
    if k is not None:
        freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===
    if L_test_scale is not None:
        freqs[k-1] = freqs[k-1] / L_test_scale

    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis

# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


mv_map = {
    0: [0, 1, 2, 3],
    1: [0, 1],
    2: [0, 2, 6],
    3: [0, 3, 4],
    4: [3, 4, 5],
    5: [4, 5, 6],
    6: [2, 5, 6],
}


BLOCK_SIZE=128
class CrossViewMask:
    def __init__(self):
        pass
    
    def _generate_view_ids(self, grid_sizes, num_views, device):
        """Generate per-token view IDs from the grid layout."""
        tokens_per_view = torch.prod(grid_sizes[0]).item() // num_views
        view_ids = torch.repeat_interleave(
            torch.arange(num_views),
            tokens_per_view
        )
        return view_ids.to(device)

    def get_view_mask(self, num_views, device):
        view_mask = torch.zeros((num_views, num_views), dtype=torch.bool, device=device)
        for target_id in range(num_views):
            source_ids = mv_map[target_id]
            for source_id in source_ids:
                view_mask[target_id, source_id] = True
        return view_mask
    
    def compute_sum_blocks(
        self,
        seq_len,
        device,
        mask=None, 
    ):
        """
        Compute token overlap counts at the block level.

        Args:
            seq_len: Sequence length.
            device: Target device.
            mask: Frame-level mask of shape (n, n).

        Returns:
            partial_blocks and full_blocks with shape (1, 1, num_blocks, num_blocks),
            where num_blocks = seq_len // BLOCK_SIZE.
        """
        def _round_up_to_multiple(x, multiple):
            return (x + multiple - 1) // multiple * multiple
        Q_LEN = _round_up_to_multiple(seq_len, BLOCK_SIZE)
        num_blocks = Q_LEN // BLOCK_SIZE

        n = mask.shape[0]
        s = seq_len // n  # Number of tokens assigned to each frame/view bucket.

        sum_blocks = torch.zeros((num_blocks, num_blocks), dtype=torch.int32, device=device)
        for q_frame_id in range(n):
            for kv_frame_id in range(n):
                if mask[q_frame_id, kv_frame_id]:
                    # Compute the token range covered by each frame.
                    q_token_start = q_frame_id * s
                    q_token_end = (q_frame_id + 1) * s
                    
                    kv_token_start = kv_frame_id * s
                    kv_token_end = (kv_frame_id + 1) * s
                    
                    # Compute the affected block range.
                    q_start_block = q_token_start // BLOCK_SIZE
                    q_end_block = min(q_token_end // BLOCK_SIZE + 1, num_blocks)
                    
                    kv_start_block = kv_token_start // BLOCK_SIZE
                    kv_end_block = min(kv_token_end // BLOCK_SIZE + 1, num_blocks)
                    
                    for q_block in range(q_start_block, q_end_block):
                        for kv_block in range(kv_start_block, kv_end_block):
                            # Accumulate the actual token overlap inside the block pair.
                            q_block_start = max(q_block * BLOCK_SIZE, q_token_start)
                            q_block_end = min((q_block + 1) * BLOCK_SIZE, q_token_end)
                            
                            kv_block_start = max(kv_block * BLOCK_SIZE, kv_token_start)
                            kv_block_end = min((kv_block + 1) * BLOCK_SIZE, kv_token_end)
                            if q_block_start < q_block_end and kv_block_start < kv_block_end:
                                overlap_q = q_block_end - q_block_start
                                overlap_kv = kv_block_end - kv_block_start
                                sum_blocks[q_block, kv_block] += overlap_q * overlap_kv

        # Split the accumulated counts into fully-covered and partially-covered blocks.
        full_blocks = sum_blocks == (BLOCK_SIZE * BLOCK_SIZE)
        partial_blocks = (sum_blocks > 0) & (sum_blocks < (BLOCK_SIZE * BLOCK_SIZE))
        partial_blocks = partial_blocks.to(dtype=torch.int8)
        full_blocks = full_blocks.to(dtype=torch.int8)

        return partial_blocks, full_blocks

    def get_block_mask(self, num_views, seq_lens, grid_sizes, device):
        seq_lens = seq_lens.to(device)
        grid_sizes = grid_sizes.to(device)

        seq_len = seq_lens[0]
        view_mask = self.get_view_mask(num_views, device)
        partial_block_mask, full_block_mask = self.compute_sum_blocks(
            seq_len=seq_len.item(),
            device=device,
            mask=view_mask,
            # BLOCK_SIZE=128,
        )
        partial_block_mask = partial_block_mask.unsqueeze(0).unsqueeze(0)
        full_block_mask = full_block_mask.unsqueeze(0).unsqueeze(0)

        view_ids = self._generate_view_ids(grid_sizes, num_views, device)
        def sparse_mask_fn(b_idx, head_idx, q_idx, kv_idx):
            # Mark indices that still fall inside the true sequence length.
            valid_indices = (q_idx < seq_len) & (kv_idx < seq_len)
            
            # Replace invalid positions with a harmless fallback index.
            safe_q_idx = torch.where(valid_indices, q_idx, 0)
            safe_kv_idx = torch.where(valid_indices, kv_idx, 0)
            
            q_view = view_ids[safe_q_idx]
            kv_view = view_ids[safe_kv_idx]
            mask_val = view_mask[q_view, kv_view]
            
            return torch.where(valid_indices, mask_val, False)

        block_mask = flex_attention._create_sparse_block_from_block_mask(
            (partial_block_mask, full_block_mask),
            sparse_mask_fn,
            seq_lengths=(seq_len.item(), seq_len.item()),
            Q_BLOCK_SIZE=BLOCK_SIZE,
            KV_BLOCK_SIZE=BLOCK_SIZE,
        )
        return block_mask


kernel_options = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_M1": 32,
    "BLOCK_N1": 64,
    "BLOCK_M2": 64,
    "BLOCK_N2": 32,
}

@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def fused_flex_attention(q, k, v, block_mask=None):
    return flex_attention.flex_attention(q, k, v, block_mask=block_mask, kernel_options=kernel_options)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype, num_views=1, cross_view_flex_attn=None, crossview_attn_type="full"):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
            k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
            v = self.v(x.to(dtype)).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # `freqs` can be either a precomputed tensor or a ProPE attention helper.
        if isinstance(freqs, torch.Tensor):
            q=rope_apply(q, grid_sizes, freqs).to(dtype)   # (b s n d)
            k=rope_apply(k, grid_sizes, freqs).to(dtype)
        else:
            q, k = freqs(
                q = q,
                k = k,
                grid_sizes=grid_sizes,
                prope_weight=1.0,  # ProPE mixing weight.
            )
            q = q.to(dtype)
            k = k.to(dtype)
        v=v.to(dtype)

        if num_views > 1:
            if crossview_attn_type=="flex":
                q = q.transpose(1, 2)  # [b, n, s, d]
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                x = cross_view_flex_attn(q, k, v)
                x = x.transpose(1, 2)
            elif crossview_attn_type=="loop":
                q = rearrange(q, 'b (nv s) ... -> b nv s ...', nv=num_views)
                k = rearrange(k, 'b (nv s) ... -> b nv s ...', nv=num_views)
                v = rearrange(v, 'b (nv s) ... -> b nv s ...', nv=num_views)
                x = []
                for target_id in range(num_views):
                    source_ids = mv_map[target_id]
                    cur_seq_lens = (seq_lens // num_views) * len(source_ids)
                    cur_x = attention(
                        q=q[:, target_id],
                        k=rearrange(k[:, source_ids], 'b n s ... -> b (n s) ...', n=len(source_ids)),
                        v=rearrange(v[:, source_ids], 'b n s ... -> b (n s) ...', n=len(source_ids)),
                        k_lens=cur_seq_lens,
                        window_size=self.window_size,
                    )
                    x.append(cur_x)
                x = torch.stack(x, dim=1)  # [b, nv, s, ...]
                x = rearrange(x, 'b nv s ... -> b (nv s) ...', nv=num_views)
            elif crossview_attn_type=="full":
                x = attention(
                    q=q,
                    k=k,
                    v=v,
                    k_lens=seq_lens,
                    window_size=self.window_size,
                )
        else:
            kwargs = {}
            if crossview_attn_type == "blockwise_causal":
                n_frames, n_height, n_width = grid_sizes[0].tolist()
                n_tokens_per_img = n_height * n_width
                temp_mask = torch.ones(n_frames, n_frames, dtype=torch.bool, device=x.device).tril(diagonal=0)
                temp_mask = rearrange(temp_mask, 'i j -> i 1 j 1')
                temp_mask = temp_mask.repeat(1, n_tokens_per_img, 1, n_tokens_per_img)
                temp_mask = rearrange(temp_mask, 'i j k l -> (i j) (k l)')
                # print('mask', 'memory_size', temp_mask.numel() * 1 / 1024 / 1024, 'MB')
                kwargs["attn_mask"] = temp_mask
            elif crossview_attn_type == "window_causal":
                n_frames, n_height, n_width = grid_sizes[0].tolist()
                n_tokens_per_img = n_height * n_width
                win = 20  # TODO: fix magic number, 80frames, 8s 
                temp_mask = torch.ones(n_frames, n_frames, dtype=torch.bool, device=x.device).tril_(-win).logical_not_().tril_(0)
                temp_mask = rearrange(temp_mask, 'i j -> i 1 j 1')
                temp_mask = temp_mask.repeat(1, n_tokens_per_img, 1, n_tokens_per_img)
                temp_mask = rearrange(temp_mask, 'i j k l -> (i j) (k l)')
                # print('mask', 'memory_size', temp_mask.numel() * 1 / 1024 / 1024, 'MB')
                kwargs["attn_mask"] = temp_mask

            x = attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size,
                **kwargs,
            )
        
        x = x.to(dtype)

        # output
        x = x.flatten(2)  # shape: [B, L, C]
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, dtype):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)

        # compute attention
        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, dtype):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img.to(dtype))).view(b, -1, n, d)
        v_img = self.v_img(context_img.to(dtype)).view(b, -1, n, d)

        img_x = attention(
            q.to(dtype), 
            k_img.to(dtype), 
            v_img.to(dtype), 
            k_lens=None
        )
        img_x = img_x.to(dtype)
        # compute attention
        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dtype=torch.float32,
        num_views=1,
        cross_view_flex_attn=None,
        crossview_attn_type="full",
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C] or [B, T, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation + e).chunk(6, dim=-2)  # [B, 6/1, C] or [B, F, 6/1, C]
        if len(e[0].shape) != 3 and len(e[0].shape) != 4:
            raise ValueError(f"invalid e shape: {[ei.shape for ei in e]}")
        if len(x.shape) != 3:
            raise ValueError(f"invalid x shape: {x.shape}")
        time_invariant_modulation = (len(e[0].shape) == 4)

        def _unflatten(x):
            return rearrange(x, "b (f h w) c -> b f (h w) c", f=grid_sizes[0][0], h=grid_sizes[0][1], w=grid_sizes[0][2]) if time_invariant_modulation else x
    
        def _flatten(x):
            return rearrange(x, "b f hw c -> b (f hw) c") if time_invariant_modulation else x

        # self-attention
        # x: [B, L, C], e[.]: [B, T, ]
        temp_x = _unflatten(self.norm1(x)) * (1 + e[1]) + e[0]
        temp_x = _flatten(temp_x)
        temp_x = temp_x.to(dtype)
        y = self.self_attn(
            temp_x,
            seq_lens,
            grid_sizes,
            freqs,
            dtype,
            num_views=num_views,
            cross_view_flex_attn=cross_view_flex_attn,
            crossview_attn_type=crossview_attn_type,
        )
        x = x + _flatten(_unflatten(y) * e[2])

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            # cross-attention
            x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype)

            # ffn function
            temp_x = _unflatten(self.norm2(x)) * (1 + e[4]) + e[3]
            temp_x = _flatten(temp_x)
            temp_x = temp_x.to(dtype)
            y = self.ffn(temp_x)
            x = x + _flatten(_unflatten(y) * e[5])
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e, grid_sizes):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C] or [B, T, C]
        """
        e = (self.modulation + e.unsqueeze(-2)).chunk(2, dim=-2)  # [B, 2/1, C] or [B, T, 2/1, C]
        if len(e[0].shape) != 3 and len(e[0].shape) != 4:
            raise ValueError(f"invalid e shape: {[ei.shape for ei in e]}")
        if len(x.shape) != 3:
            raise ValueError(f"invalid x shape: {x.shape}")
        time_invariant_modulation = (len(e[0].shape) == 4)

        def _unflatten(x):
            return rearrange(x, "b (f h w) c -> b f (h w) c", f=grid_sizes[0][0], h=grid_sizes[0][1], w=grid_sizes[0][2]) if time_invariant_modulation else x
          
        def _flatten(x):
            return rearrange(x, "b f hw c -> b (f hw) c") if time_invariant_modulation else x

        x = (self.head(_flatten(_unflatten(self.norm(x)) * (1 + e[1]) + e[0])))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        add_plucker_fourier=False,
        enable_prope_emb=False,
        in_dim_control_adapter=24,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        add_embedding=False,
        # plucker_embedding=False,
        hdmap_embedding=False,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        
        # normal embedding
        if add_embedding:
            self.add_embedding = True
            self.add_conv = nn.Conv3d(
                16, dim,
                kernel_size=patch_size, stride=patch_size)
            self.add_proj = zero_module(nn.Linear(dim, dim))
            nn.init.xavier_uniform_(self.add_conv.weight.flatten(1))
        else:
            self.add_embedding = False
        
        if hdmap_embedding:
            self.hdmap_embedding = True
            self.hdmap_conv = nn.Conv3d(
                16, dim,
                kernel_size=patch_size, stride=patch_size)
            self.hdmap_proj = zero_module(nn.Linear(dim, dim))
            nn.init.xavier_uniform_(self.hdmap_conv.weight.flatten(1))
        else:
            self.hdmap_embedding = False

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)
        self.enable_prope_emb = enable_prope_emb

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d
        self.dim = dim
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1
        )

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
        
        self.control_adapter = None

        if add_ref_conv:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None

        self.teacache = None
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None
        self.gradient_checkpointing = False
        self.sp_world_size = 1
        self.sp_world_rank = 0
    
    def enable_teacache(
        self,
        coefficients,
        num_steps: int,
        rel_l1_thresh: float,
        num_skip_start_steps: int = 0,
        offload: bool = True
    ):
        self.teacache = TeaCache(
            coefficients, num_steps, rel_l1_thresh=rel_l1_thresh, num_skip_start_steps=num_skip_start_steps, offload=offload
        )

    def disable_teacache(self):
        self.teacache = None

    @enable_cfg_skip()
    def enable_cfg_skip(self, cfg_skip_ratio, num_steps):
        if cfg_skip_ratio != 0:
            self.cfg_skip_ratio = cfg_skip_ratio
            self.current_steps = 0
            self.num_inference_steps = num_steps
        else:
            self.cfg_skip_ratio = None
            self.current_steps = 0
            self.num_inference_steps = None

    @disable_cfg_skip()
    def disable_cfg_skip(self):
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None

    def enable_riflex(
        self,
        k = 6,
        L_test = 66,
        L_test_scale = 4.886,
    ):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                get_1d_rotary_pos_embed_riflex(1024, self.d - 4 * (self.d // 6), use_real=False, k=k, L_test=L_test, L_test_scale=L_test_scale),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def disable_riflex(self):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                rope_params(1024, self.d - 4 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def enable_multi_gpus_inference(self,):
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn)

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value
    
    def get_cross_view_flex_attn(self, seq_lens, grid_sizes, num_views, device):
        current_config= {
            'seq_len': seq_lens[0].item(),
        }
        if hasattr(self, 'cross_view_flex_attn') and hasattr(self, 'current_config') and self.current_config == current_config:
            return self.cross_view_flex_attn
        else:
            block_mask_fn = CrossViewMask()
            block_mask = block_mask_fn.get_block_mask(
                num_views, seq_lens, grid_sizes, device)
            self.cross_view_flex_attn = functools.partial(fused_flex_attention, block_mask=block_mask)
            self.current_config = current_config
        return self.cross_view_flex_attn

    @cfg_skip()
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        y_camera=None,
        y_add=None,
        full_ref=None,
        cond_flag=True,
        enable_view_emb=False,
        num_views=1,
        plucker_embs=None,
        P_mats=None,
        P_inv_mats=None,
        hdmap_emb=None,
        dtype=None,
        crossview_attn_type="full",
        step=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B] or [B, F]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            cond_flag (`bool`, *optional*, defaults to True):
                Flag to indicate whether to forward the condition input

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        if len(t.shape) != 1 and len(t.shape) != 2:
            raise ValueError(f"t must be of shape [B] or [B, F], but got {t.shape}")

        # if self.training:
        #     patch_embedding_weight_mean = self.patch_embedding.weight.abs().mean(dim=[0, 2, 3, 4])
        #     logger.info(f"patch_embedding_weight_mean: {patch_embedding_weight_mean}")

        # params
        device = self.patch_embedding.weight.device
        if dtype is None:
            dtype = x.dtype
        
        if self.freqs.device != device and torch.device(type="meta") != device:
            if isinstance(self.freqs, torch.Tensor):
                self.freqs = self.freqs.to(device)
            else:
                self.freqs.to(device)
        
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]   # (1, d, nf, h, w)

        if enable_view_emb:
            # Inject the view index as an additional embedding.
            _, d, nf, h, w = x[0].shape
            view_ids = torch.range(0, num_views - 1, device=device).float()  #(n, )
            views_emb = sinusoidal_embedding_batchwise(d, view_ids).to(dtype)  # (n, d)
            views_emb = views_emb.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (1, n, d, 1, 1, 1)
            # repeat views_emb to match the x[0] shape
            views_emb = views_emb.repeat(1, 1, 1, nf // num_views, h, w)  # (1, n, d, f, h, w)
            views_emb = rearrange(views_emb, 'b n d f h w -> b d (n f) h w')
            x = [u + views_emb for u in x]  # (1, d, nf, h, w)

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

        x = [u.flatten(2).transpose(1, 2) for u in x]

        # add control adapter
        if self.control_adapter is not None and y_camera is not None:
            y_camera = self.control_adapter(y_camera)
            y_camera = self.pose_control_proj(y_camera.flatten(2).transpose(1, 2))
            x = [u + v for u, v in zip(x, y_camera)]
        
        # add embedding
        if self.add_embedding and y_add is not None:
            y_add = self.add_conv(y_add)
            y_add = self.add_proj(y_add.flatten(2).transpose(1, 2))
            x = [u + v for u, v in zip(x, y_add)]
        
        # if self.plucker_embedding and plucker_embs is not None:
        #     # plucker_embs = plucker_embed_with_fourier(plucker_embs, self.plucker_embedder_obj)
        #     plucker_embs = self.plucker_conv(plucker_embs)
        #     plucker_embs = self.plucker_proj(plucker_embs.flatten(2).transpose(1, 2))
        #     x = [u + v for u, v in zip(x, plucker_embs)]
        
        if self.hdmap_embedding and hdmap_emb is not None:
            hdmap_emb = self.hdmap_conv(hdmap_emb)
            hdmap_emb = self.hdmap_proj(hdmap_emb.flatten(2).transpose(1, 2))
            x = [u + v for u, v in zip(x, hdmap_emb)]
        
        if self.enable_prope_emb and P_mats is not None and P_inv_mats is not None:
            self.freqs.set_projection_matrices(P_mats, P_inv_mats)  # Set the projection matrices for the rotary embeddings
        
        if self.ref_conv is not None and full_ref is not None:
            full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += full_ref.size(1)
            x = [torch.concat([_full_ref.unsqueeze(0), u], dim=1) for _full_ref, u in zip(full_ref, x)]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        # time embeddings
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_batchwise(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(-1, (6, self.dim))  # [B, 6, C] or [B, F, 6, C]

            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            # e0 = e0.to(dtype)
            # e = e.to(dtype)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        
        # Build the sparse block mask for flexible cross-view attention.
        if num_views > 1 and crossview_attn_type == 'flex':
            cross_view_flex_attn = self.get_cross_view_flex_attn(
                seq_lens, grid_sizes, num_views, device)
        else:
            cross_view_flex_attn = None
        
        if self.enable_prope_emb and step is not None:
            # Ramp the ProPE weight from 0 to 1 between steps 50 and 600.
            prope_weight = max(0.0, min(1.0, (step-50.0) / (600-50.0)))
            self.freqs.set_prope_weight(prope_weight)  # Set the prope weight for the rotary embeddings
            if step % 50 == 0:
                logger.info(f"Step {step}: Prope weight set to {prope_weight:.4f}", main_process_only=True)

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
        # TeaCache
        if self.teacache is not None:
            if cond_flag:
                modulated_inp = e0
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if skip_flag:
                    self.should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input, modulated_inp)
                        self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        self.should_calc = False
                    else:
                        self.should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.should_calc = self.should_calc
            else:
                self.should_calc = self.teacache.should_calc
        
        # TeaCache
        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            e0,
                            seq_lens,
                            grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            dtype,
                            num_views,
                            cross_view_flex_attn,
                            crossview_attn_type,
                            **ckpt_kwargs,
                        )
                    else:
                        # arguments
                        kwargs = dict(
                            e=e0,
                            seq_lens=seq_lens,
                            grid_sizes=grid_sizes,
                            freqs=self.freqs,
                            context=context,
                            context_lens=context_lens,
                            dtype=dtype,
                            num_views=num_views,
                            cross_view_flex_attn=cross_view_flex_attn,
                            crossview_attn_type=crossview_attn_type,
                        )
                        x = block(x, **kwargs)
                    
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        e0,
                        seq_lens,
                        grid_sizes,
                        self.freqs,
                        context,
                        context_lens,
                        dtype,
                        num_views,
                        cross_view_flex_attn,
                        crossview_attn_type,
                        **ckpt_kwargs,
                    )
                else:
                    # arguments
                    kwargs = dict(
                        e=e0,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        freqs=self.freqs,
                        context=context,
                        context_lens=context_lens,
                        dtype=dtype,
                        num_views=num_views,
                        cross_view_flex_attn=cross_view_flex_attn,
                        crossview_attn_type=crossview_attn_type,
                    )
                    x = block(x, **kwargs)

        if self.sp_world_size > 1:
            x = get_sp_group().all_gather(x, dim=1)

        if self.ref_conv is not None and full_ref is not None:
            full_ref_length = full_ref.size(1)
            x = x[:, full_ref_length:]
            grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        # head
        x = self.head(x, e, grid_sizes)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        logger.info(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]
        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                from diffusers.models.modeling_utils import \
                    load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available
                if is_accelerate_available():
                    import accelerate
                
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(model_file_safetensors)
                else:
                    from safetensors.torch import load_file, safe_open
                    model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                    state_dict = {}
                    logger.info(model_files_safetensors)
                    for _model_file_safetensors in model_files_safetensors:
                        _state_dict = load_file(_model_file_safetensors)
                        for key in _state_dict:
                            state_dict[key] = _state_dict[key]

                if diffusers_version >= "0.33.0":
                    # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
                    # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
                    load_model_dict_into_meta(
                        model,
                        state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )
                else:
                    model._convert_deprecated_attention_blocks(state_dict)
                    # move the params from meta device to cpu
                    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                    if len(missing_keys) > 0:
                        raise ValueError(
                            f"Cannot load {cls} from {pretrained_model_path} because the following keys are"
                            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                            " those weights or else make sure your checkpoint file is correct."
                        )

                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        logger.info(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )
                
                return model
            except Exception as e:
                logger.warning(
                    f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead."
                )
        
        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]
        
        if model.state_dict()['patch_embedding.weight'].size() != state_dict['patch_embedding.weight'].size():
            model.state_dict()['patch_embedding.weight'][:, :state_dict['patch_embedding.weight'].size()[1], :, :] = state_dict['patch_embedding.weight']
            model.state_dict()['patch_embedding.weight'][:, state_dict['patch_embedding.weight'].size()[1]:, :, :] = 0
            state_dict['patch_embedding.weight'] = model.state_dict()['patch_embedding.weight']
        
        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                logger.warning(f"{key} Size don't match, skip")
                
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        logger.warning(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)
        
        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")
        
        model = model.to(torch_dtype)
        return model
