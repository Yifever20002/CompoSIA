import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image
from torchvision.io import read_image
from transformers import T5Tokenizer

from ..models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                              WanT5EncoderModel, UnifiedTransformer3DModel)
from ..utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                get_sampling_sigmas)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from composia.utils.train_utils import prepare_clip_context, prepare_mask_condition, batch_encode_vae, prepare_action_condition, sample_proj_mats, get_consistent_texture_video

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    videos: torch.Tensor


class WanFunUnifiedPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->clip_image_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: UnifiedTransformer3DModel,
        clip_image_encoder: CLIPModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, clip_image_encoder=clip_image_encoder, scheduler=scheduler
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.config.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.config.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.config.spatial_compression_ratio, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds


    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, num_cameras, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            ((num_frames - 1) // self.vae.config.temporal_compression_ratio + 1) * num_cameras,
            height // self.vae.config.spatial_compression_ratio,
            width // self.vae.config.spatial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents


    def decode_latents(self, latents: torch.Tensor, num_cameras=1) -> torch.Tensor:
        latents = latents.to(self.vae.dtype)
        latents = rearrange(latents, "b c (n f) h w -> (b n) c f h w", n=num_cameras)
        frames = self.vae.decode(latents).sample
        frames = rearrange(frames, "(b n) c f h w -> b c (n f) h w", n=num_cameras)
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.cpu().float().numpy()
        return frames
    

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt
    
    def convert_rgb_mask_to_latent_mask(self, mask: torch.Tensor, temporal_compression_ratio=4, spatial_downsample=8, dilate=False) -> torch.Tensor:
        """
        Convert a per-frame mask [T, 1, H, W] to latent resolution [1, T_latent, 1, H', W'].
        T_latent groups frames by the temporal VAE downsample factor k = vae_scale_factor_temporal:
        [0], [1..k], [k+1..2k], ...
        """

        k = temporal_compression_ratio
        mask0 = mask[0:1]  # [1,1,H,W]
        mask1 = mask[1::k]  # [T'-1,1,H,W]
        sampled = torch.cat([mask0, mask1], dim=0)  # [T',1,H,W]
        pooled = sampled.permute(1, 0, 2, 3).unsqueeze(0)

        # Convert image-space masks to the transformer patch grid in one step.
        spatial_downsample = 8
        patch_size = 2
        eff_down = spatial_downsample * patch_size

        if dilate:
            pooled_patch = F.max_pool3d(
                pooled,
                kernel_size=(1, eff_down, eff_down),
                stride=(1, eff_down, eff_down),
                padding=0,
            )
        else:
            pooled_patch = F.avg_pool3d(
                pooled,
                kernel_size=(1, eff_down, eff_down),
                stride=(1, eff_down, eff_down)
            )

        pooled_patch = pooled_patch.to(mask.dtype)

        _, _, T_lat, H_patch, W_patch = pooled_patch.shape

        mask_latent = pooled_patch \
            .repeat_interleave(patch_size, dim=-2) \
            .repeat_interleave(patch_size, dim=-1)

        H_lat = H_patch * patch_size
        W_lat = W_patch * patch_size
        mask_latent = mask_latent[..., :H_lat, :W_lat]

        return mask_latent

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        mask: Union[torch.FloatTensor] = None,
        mask_video: Union[torch.FloatTensor] = None,
        video: Union[torch.FloatTensor] = None,
        num_frames: int = 49,
        num_cameras: int = 1,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        clip_image: Union[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        comfyui_progressbar: bool = False,
        shift: int = 5,
        additional_conditions: Dict = {},
        crossview_attn_type: str = "full",
        step=None,
        ttm=None,
        ttm_origin=None,
        use_t_variant_noise: bool = False,
        id_injection_level: float = 0.6,
        t_compression_ratio: int = 1,  # 4 for Wan-Fun
        num_unroll_steps: int = 1,
        num_condition_images: int = 1,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.
        Args:
        video: (b, nf, c, h, w) tensor of input video.
        mask_video: (b, nf, 1, h, w)

        Examples:


        Returns:

        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else:
            in_prompt_embeds = prompt_embeds
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler,
                device=device,
                sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 2)

        latent_channels = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            num_cameras,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )
        dtype = latents.dtype
        if comfyui_progressbar:
            pbar.update(1)

        if mask is not None and mask_video is not None:
            mask = mask.to(dtype=torch.float32, device=device)
            mask_video = mask_video.to(dtype=dtype, device=device)
            mask_latents = prepare_mask_condition(mask, latents, num_cameras=num_cameras, temporal_compression_ratio=self.vae.config.temporal_compression_ratio)
            masked_video_latents = batch_encode_vae(mask_video, self.vae, num_cameras=num_cameras)
        else:
            mask_latents = None
            masked_video_latents = None

        video = video.to(dtype=dtype, device=device)
        video_latents = batch_encode_vae(video, self.vae, num_cameras=num_cameras)

        
        for name, value in additional_conditions.items():
            if name == "bbox":
                value = value.to(dtype=dtype, device=device)
                value = batch_encode_vae(value, self.vae, num_cameras=num_cameras)
                additional_conditions[name] = value.to(dtype=dtype, device=device)
        
        tstrong = 1e5
        ttm_latents = None
        mask_erosion = None
        mask_dilate = None
        mask_ref_origin = None
        if ttm is not None:
            id_injection_level = max(0.0, min(1.0, id_injection_level))
            tstrong_idx = min(int(id_injection_level * len(timesteps)), len(timesteps) - 1)
            tstrong = timesteps[tstrong_idx]
            mag = ttm.abs().sum(dim=1, keepdim=True)
            mask_ref = (mag > 1e-6).float()
            tex_raw = read_image("assets/bg/nature_texture.png")
            tex_big = tex_raw.to(device=ttm.device, dtype=ttm.dtype) / 255.0
            texture_bg = get_consistent_texture_video(tex_big, ttm.shape)
            ttm = ttm * mask_ref + (1 - mask_ref) * texture_bg
            ttm = ttm * 2.0 - 1.0
            ttm = ttm.to(dtype=dtype, device=device).unsqueeze(0)
            ttm_latents = batch_encode_vae(ttm, self.vae, num_cameras=num_cameras)
            # Use eroded and dilated latent masks to blend the edited object
            # while protecting the context around it during denoising.
            mask_erosion = self.convert_rgb_mask_to_latent_mask(mask_ref, temporal_compression_ratio=self.vae.config.temporal_compression_ratio, dilate=False).to(dtype=dtype, device=device)
            torch_rng = torch.Generator(device=device)

            if ttm_origin is not None:
                mag_origin = ttm_origin.abs().sum(dim=1, keepdim=True)
                mask_ref_origin = (mag_origin > 1e-6).float()
                mask_union = ((mask_ref > 0) | (mask_ref_origin > 0)).float()
                mask_dilate = self.convert_rgb_mask_to_latent_mask(mask_union, temporal_compression_ratio=self.vae.config.temporal_compression_ratio, dilate=True).to(dtype=dtype, device=device)

                del ttm_origin, mag_origin, mask_union
            else:
                mask_dilate = self.convert_rgb_mask_to_latent_mask(mask_ref, temporal_compression_ratio=self.vae.config.temporal_compression_ratio, dilate=True).to(dtype=dtype, device=device)

            del mask_ref
            torch.cuda.empty_cache()

        if "proj_mats" in additional_conditions:
            proj_mats = additional_conditions.pop("proj_mats")
            proj_mats = sample_proj_mats(proj_mats, num_cameras=num_cameras)      
        else:
            proj_mats = None

        if "action" in additional_conditions:
            additional_conditions["action"] = prepare_action_condition(additional_conditions["action"], temporal_compression_ratio=self.vae.config.temporal_compression_ratio).to(dtype=dtype, device=device)

        if clip_image is not None and self.clip_image_encoder is not None:
            clip_context = prepare_clip_context(self.clip_image_encoder, clip_image, device=device, dtype=weight_dtype)
        else:
            clip_context = None

        if comfyui_progressbar:
            pbar.update(1)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        target_shape = (self.vae.config.latent_channels, ((num_frames - 1) // self.vae.config.temporal_compression_ratio + 1)*num_cameras, width // self.vae.config.spatial_compression_ratio, height // self.vae.config.spatial_compression_ratio)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1]) 
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps

        if mask_latents is not None and masked_video_latents is not None:
            if not use_t_variant_noise:
                mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                )
                y = torch.cat([mask_input, masked_video_latents_input], dim=1).to(device, weight_dtype)
            else:
                assert len(mask_latents.shape) == 5, "mask_latents should be a 5D tensor"
                fill_mask = (mask_latents[:, :1, :, :, :] > 0).repeat(1, latents.shape[1], 1, 1, 1)
                if ttm_latents is not None:
                    fill_mask = fill_mask & (mask_dilate < 0.5)
                latents[fill_mask] = masked_video_latents[fill_mask].to(latents.dtype)
                y = None
        else:
            y = None

        if clip_context is not None: 
            clip_context_input = (
                torch.cat([clip_context] * 2) if do_classifier_free_guidance else clip_context
            )
        else:
            clip_context_input = None

        for name, value in additional_conditions.items():
            if do_classifier_free_guidance:
                additional_conditions[name] = torch.cat([value] * 2)

        final_video = None
        with self.progress_bar(total=num_inference_steps * num_unroll_steps) as progress_bar:
            for unroll_step in range(num_unroll_steps):
                for i, t in enumerate(timesteps):
                    self.transformer.current_steps = i
                    if self.interrupt:
                        continue
                    if use_t_variant_noise and mask_latents is not None:
                        latents[fill_mask] = masked_video_latents[fill_mask].to(latents.dtype)

                    if t > tstrong:
                        sigma = self.scheduler.sigmas[i].to(device=latents.device, dtype=latents.dtype)
                        ttm_noise = torch.randn(ttm_latents.size(), device=ttm_latents.device, generator=torch_rng, dtype=dtype)
                        noisy_ttm = (1.0 - sigma) * ttm_latents + sigma * ttm_noise
                        latents = mask_erosion * noisy_ttm + (1 - mask_erosion) * latents
                    
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    if hasattr(self.scheduler, "scale_model_input"):
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    timestep = t.expand(latent_model_input.shape[0])
                    if use_t_variant_noise and mask_latents is not None:
                        if ttm_latents is not None:
                            timestep = timestep[:, None, None, None].repeat(1, target_shape[1], target_shape[3], target_shape[2]).contiguous()
                            timestep_fill_mask = mask_latents[:, 0, :, :, :] > 0
                            timestep_fill_mask = timestep_fill_mask & (mask_dilate[:, 0, :, :, :] < 0.5)
                            if do_classifier_free_guidance:
                                timestep_fill_mask = torch.cat([timestep_fill_mask] * 2, dim=0)
                            timestep[timestep_fill_mask] = timesteps[-1]

                            timestep = F.avg_pool3d(timestep.unsqueeze(1).float(),
                                                    kernel_size=(1, 2, 2),
                                                    stride=(1, 2, 2),
                                                ).squeeze(1).to(timestep.dtype)

                        else:
                            timestep = timestep.unsqueeze(-1).repeat(1, target_shape[1]).contiguous()
                            timestep_fill_mask = mask_latents[:, 0, :, 0, 0] > 0

                            if do_classifier_free_guidance:
                                timestep_fill_mask = torch.cat([timestep_fill_mask] * 2, dim=0)
                            timestep[timestep_fill_mask] = timesteps[-1]
                    else:
                        timestep = timestep.unsqueeze(-1).repeat(1, target_shape[1]).contiguous()
                    with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                        noise_pred, _ = self.transformer(
                            x=latent_model_input,
                            context=in_prompt_embeds,
                            t=timestep,
                            seq_len=seq_len,
                            y=y,
                            clip_fea=clip_context_input,
                            num_views=num_cameras,
                            proj_mats=proj_mats,
                            dtype=dtype,
                            additional_conditions=additional_conditions,
                            crossview_attn_type=crossview_attn_type,
                            step=step,
                        )
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                        
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                    if comfyui_progressbar:
                        pbar.update(1)

                torch.cuda.empty_cache()
                if output_type == "numpy":
                    video = self.decode_latents(latents, num_cameras=num_cameras)
                elif not output_type == "latent":
                    video = self.decode_latents(latents, num_cameras=num_cameras)
                    video = self.video_processor.postprocess_video(video=video, output_type=output_type)
                else:
                    video = latents

                if unroll_step == 0:
                    final_video = video
                else:
                    final_video = np.concatenate([final_video, video[:, :, num_condition_images:, :, :]], axis=2)

                if unroll_step < num_unroll_steps - 1:
                    assert y is None, "y should be None for unroll_step > 0"
                    assert proj_mats is None, "proj_mats should be None for unroll_step > 0"
                    
                    num_cond_latents = (num_condition_images - 1) // t_compression_ratio + 1
                    last_latents = latents.clone()
                    last_cond_latents = last_latents[:, :, -num_cond_latents:, :, :].clone()
                    latents = randn_tensor(last_latents.shape, generator=generator, device=device, dtype=weight_dtype)

                    last_cond_latents = batch_encode_vae(torch.from_numpy(video[:, :, -num_condition_images:]).cuda().permute(0, 2, 1, 3, 4) * 2 - 1, self.vae, num_cameras=num_cameras)
                    latents[:, :, :num_cond_latents, :, :] = last_cond_latents
                    self.scheduler._step_index = None

                    assert use_t_variant_noise, "use_t_variant_noise should be True for unroll_step > 0"
                    masked_video_latents = latents.clone()
                    mask_latents = torch.zeros_like(latents)
                    mask_latents[:, :, :num_cond_latents, :, :] = 1
                    fill_mask = (mask_latents[:, :1, :, :, :] > 0).repeat(1, latents.shape[1], 1, 1, 1)

        self.maybe_free_model_hooks()

        if not return_dict:
            final_video = torch.from_numpy(final_video)

        return WanPipelineOutput(videos=final_video)
