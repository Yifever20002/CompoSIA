from typing import Any
from torch import nn

from pydantic import BaseModel
from videox_fun.utils.logger import logger


class Components(BaseModel):
    # pipeline cls
    pipeline_cls: Any = None

    # Tokenizers
    tokenizer: Any = None
    tokenizer_2: Any = None
    tokenizer_3: Any = None

    # Text encoders
    text_encoder: Any = None
    text_encoder_2: Any = None
    text_encoder_3: Any = None

    # Image encoders
    clip_image_encoder: Any = None
    image_encoder_2: Any = None
    image_encoder_3: Any = None

    # Autoencoder
    vae: Any = None

    # Denoiser
    transformer3d: Any = None
    unet: Any = None

    # Scheduler
    noise_scheduler: Any = None

    # VGGT
    vggt_aggregator: Any = None

    # patch_embed
    patch_embed: Any = None
    # projectors 
    # angular_proj: Any = None
    # scale_proj: Any = None

    def print_components(self):
        for k, v in self.__dict__.items():
            if isinstance(v, nn.Module):
                device = None
                dtype = None
                if hasattr(v, "device"):
                    device = v.device
                if hasattr(v, "dtype"):
                    dtype = v.dtype
                logger.info(f"[component module] {k}: {type(v)}, device: {device}, dtype: {dtype}")
