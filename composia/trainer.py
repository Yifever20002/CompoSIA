"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
import logging
import os
import sys
import json
import math
from typing import Union

import accelerate
import diffusers
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import check_min_version
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from transformers.utils import ContextManagers
from typing_extensions import override
from einops import rearrange
import datasets
from copy import deepcopy

from videox_fun.utils.config import Config

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from torch.utils.data import DataLoader, Dataset
from composia.schemas.components import Components
from composia.schemas.state import State
from composia.utils.memory_utils import get_memory_statistics, free_memory
from composia.utils.metric_utils import (
    evaluate_met3r,
    StyleGanFVDMetric,
    FIDDistMetric,
    compute_vbench_metric,
    VbenchScore,
)
from composia.utils.train_utils import WAN_FUN_NEGATIVE_PROMPT
from videox_fun.models import CLIPModel, WanT5EncoderModel, UnifiedTransformer3DModel
from videox_fun.utils.utils import (
    construct_emb_cls,
    filter_kwargs,
    import_cls,
    save_met3r_debug_images,
    save_multiview_videos_grid,
)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

LOG_NAME = "trainer"
LOG_LEVEL = "INFO"
logger = get_logger(LOG_NAME, LOG_LEVEL)

class Trainer:
    def __init__(self, args):
        self.args = args
        self.state: State = State()
        self.config = Config.fromfile(self.args.config_path).to_omegaconf()
        # load args from config
        if self.config.get("args", None) is not None:
            for key, value in self.config.args.items():
                if key not in self.args:
                    setattr(self.args, key, value)
        
        # update args with config trainer_kwargs
        if self.config.get("trainer_kwargs", None) is not None:
            for key, value in self.config.trainer_kwargs.items():
                setattr(self.args, key, value)
        
        self.components = Components()
        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories()
        self.components = self.load_components()
        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    def _init_distributed(self):
        args = self.args
        # logging_dir = "/job_tboard"
        logging_dir = os.path.join(args.output_dir, args.logging_dir)
        # print('logging_dir', logging_dir)
        os.makedirs(logging_dir, exist_ok=True)
        save_config_path = os.path.join(logging_dir, "model_param.yaml")
        with open(save_config_path, 'w') as f:
            OmegaConf.save(self.config, f)

        accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )
        
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        # deepspeed_plugin = None
        if deepspeed_plugin is not None:
            self.state.zero_stage = int(deepspeed_plugin.zero_stage)
            print(f"Using DeepSpeed Zero stage: {self.state.zero_stage}")
        else:
            self.state.zero_stage = 0
            print("DeepSpeed is not enabled.")

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)
            self.state.rng = np.random.default_rng(np.random.PCG64(args.seed + self.accelerator.process_index))
            self.state.torch_rng = torch.Generator(self.accelerator.device).manual_seed(args.seed + self.accelerator.process_index)
        else:
            self.state.rng = None
            self.state.torch_rng = None
        self.state.index_rng = np.random.default_rng(np.random.PCG64(43))
        print(f"Init rng with seed {args.seed + self.accelerator.process_index}. Process_index is {self.accelerator.process_index}")

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.state.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.state.weight_dtype = torch.float16
            self.args.mixed_precision = self.accelerator.mixed_precision
        elif self.accelerator.mixed_precision == "bf16":
            self.state.weight_dtype = torch.bfloat16
            self.args.mixed_precision = self.accelerator.mixed_precision


    def _init_logging(self):
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()


    def _init_directories(self):
        # Handle the repository creation
        if self.accelerator.is_main_process:
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)

    def prepare_trackers(self):
        if self.accelerator.is_main_process:
            tracker_config = dict(vars(self.args))
            tracker_config.pop("val_data_meta", None)
            tracker_config.pop("trainable_modules", None)
            tracker_config.pop("trainable_modules_low_learning_rate", None)
            
            # Keep only tracker config values supported by the logger backend.
            tracker_config_tmp = deepcopy(tracker_config)
            for key, value in tracker_config.items():
                if not isinstance(value, (int, float, str, bool, torch.Tensor)):
                    tracker_config_tmp.pop(key)
            
            self.accelerator.init_trackers(self.args.tracker_project_name, tracker_config_tmp)
    
    def eval(self):
        self.prepare_for_validation(clip_validation_set=False)
        self.prepare_trackers()
        self.validate("final", compute_vbench=False)

    def prepare_for_validation(self, clip_validation_set=True):
        raise NotImplementedError

    def validate(self, step: Union[int, str], compute_vbench=False) -> None:
        raise NotImplementedError

    def load_components(self):
        raise NotImplementedError

    def initialize_pipeline(self):
        raise NotImplementedError


class WanUnifiedTrainer(Trainer):
    @override
    def load_components(self):
        components = Components()
        config = self.config
        args = self.args

        components.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )

        components.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )
        print("TORCH_HOME in python:", os.getenv("TORCH_HOME"))
        print("torch.hub.get_dir():", torch.hub.get_dir())

        components.transformer3d = UnifiedTransformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        ).to(self.state.weight_dtype)

        if "T2V" not in args.pretrained_model_name_or_path:
            components.clip_image_encoder = CLIPModel.from_pretrained(
                os.path.join(args.pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
            )
            components.clip_image_encoder = components.clip_image_encoder.eval()

        if args.transformer_path is not None:
            print(f"From checkpoint: {args.transformer_path}")

            if args.transformer_path.endswith("safetensors"):
                # Support both Wan safetensors checkpoints and legacy .pt exports.
                from safetensors.torch import load_file
                state_dict = load_file(args.transformer_path)
            else:
                ckpt = torch.load(args.transformer_path, map_location="cpu")

                if "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                elif "module" in ckpt:
                    state_dict = ckpt["module"]
                else:
                    state_dict = ckpt

            model_state_dict = components.transformer3d.state_dict()
            filtered_state_dict = {}
            skipped_keys = []
            for key, value in state_dict.items():
                # Older checkpoints may still contain branches removed from the
                # open-source eval model. Load only keys that still exist and match shape.
                if key in model_state_dict and model_state_dict[key].size() == value.size():
                    filtered_state_dict[key] = value
                else:
                    skipped_keys.append(key)

            if len(skipped_keys) > 0:
                print(f"skip {len(skipped_keys)} checkpoint keys not used by current model")
                skipped_prefixes = sorted({key.split(".")[0] for key in skipped_keys})
                print(f"skipped key prefixes: {skipped_prefixes}")
                print("first skipped keys:")
                for key in skipped_keys[:50]:
                    print(f"  {key}")

            m, u = components.transformer3d.load_state_dict(filtered_state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        def deepspeed_zero_init_disabled_context_manager():
            """
            Return a context manager list that disables zero.Init when needed.
            """
            deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
            if deepspeed_plugin is None:
                return []

            return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            components.text_encoder = WanT5EncoderModel.from_pretrained(
                os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
                additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
                low_cpu_mem_usage=True,
                torch_dtype=self.state.weight_dtype,
            )
            components.text_encoder = components.text_encoder.eval()
            if args.vae_path is not None:
                vae_path = args.vae_path
            else:
                vae_path = os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae'))
            AutoencoderKL = import_cls(config['vae_kwargs'].get('vae_type', 'videox_fun.models.AutoencoderKLWan'))
            components.vae = AutoencoderKL.from_pretrained(
                vae_path,
                additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
            )

        components.print_components()

        return components

    @override
    def initialize_pipeline(self):
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(self.config['scheduler_kwargs']))
        )
        if self.args.low_vram and self.args.train_mode != "normal" and self.components.clip_image_encoder is not None:
            self.components.clip_image_encoder.to(self.accelerator.device)
        pipeline = import_cls(self.config['pipeline'])(
            vae=self.accelerator.unwrap_model(self.components.vae).to(self.state.weight_dtype),
            text_encoder=self.accelerator.unwrap_model(self.components.text_encoder),
            tokenizer=self.components.tokenizer,
            transformer=self.accelerator.unwrap_model(self.components.transformer3d),
            scheduler=scheduler,
            clip_image_encoder=self.components.clip_image_encoder,
        )

        return pipeline

    def prepare_for_validation(self, clip_validation_set=True):
        """
        Prepare validation data by loading metadata file and extracting first frames.
        """
        validation_kwargs = self.config.get("validation_kwargs", {})
        dataset_kwargs = validation_kwargs.get("dataset_kwargs", {})
        val_datasets = {}
        for dataset_name, dataset_config in dataset_kwargs.items():
            val_datasets[dataset_name] = construct_emb_cls(dataset_config)
        self.val_datasets = val_datasets

        self.val_modes = validation_kwargs.get("val_modes", {})
        self.eval_metrics = validation_kwargs.get("eval_metrics", ["fid"])
        self.max_validation_samples = validation_kwargs.get("max_validation_samples", 1) * self.accelerator.num_processes

    @torch.no_grad()
    def validate(self, step: Union[int, str], compute_vbench=False) -> None:
        if len(self.val_modes) == 0:
            logger.warning("No validation modes configured. Skipping validation.")
            return

        if step != "final" and len(self.val_modes) > 1:
            sample_names = []
            for name, cfg in self.val_modes.items():
                every_n = cfg.get("every_n", 1)
                if step % (self.args.validation_steps * every_n) == 0:
                    sample_names.append(name)
        else:
            sample_names = list(self.val_modes.keys())

        logger.info(f"Validating {len(sample_names)} modes: {sample_names}...")

        self.components.transformer3d.eval()
        torch.set_grad_enabled(False)

        pipeline = self.initialize_pipeline()
        pipeline = pipeline.to(self.accelerator.device)
        logger.info(f"Process {self.accelerator.process_index} Pipeline initialized")

        for i, sample_name in enumerate(sample_names):
            dataset_name = self.val_modes[sample_name]["dataset_name"]
            dataset_kwargs = self.val_modes[sample_name].get("kwargs", {})
            logger.info(f"Validating {i+1}/{len(sample_names)} mode: {sample_name}... dataset_name: {dataset_name}, dataset_kwargs: {dataset_kwargs}")
            self.validate_one_mode(
                pipeline=pipeline,
                validation_mode=sample_name,
                val_dataset=self.val_datasets[dataset_name],
                val_dataset_kwargs=dataset_kwargs,
                step=step,
                max_samples=self.max_validation_samples,
                pipeline_kwargs=self.val_modes[sample_name].get("pipeline_kwargs", {}),
                compute_vbench=compute_vbench
            )

        del pipeline
        if self.args.low_vram:
            if self.args.train_mode != "normal" and self.components.clip_image_encoder is not None:
                self.components.clip_image_encoder.to('cpu')
            self.components.vae.to('cpu')
            self.components.text_encoder.to('cpu')

        torch.set_grad_enabled(True)
        self.components.transformer3d.train()

        free_memory()

    @torch.no_grad()
    def validate_one_mode(
        self,
        pipeline,
        validation_mode: str,
        val_dataset,
        val_dataset_kwargs,
        pipeline_kwargs,
        step: Union[int, str],
        max_samples: int,
        compute_vbench: bool = False,
    ) -> None:
        """
        Perform validation using pre-extracted images with distributed processing.
        """
        accelerator = self.accelerator
        weight_dtype = self.state.weight_dtype
        logger.info(f"Process {accelerator.process_index} Starting validation...")
        free_memory()

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        if self.args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(self.args.seed)

        validation_path = os.path.join(self.args.output_dir, f"validation_res_{step}", validation_mode)
        if accelerator.is_main_process:
            os.makedirs(validation_path, exist_ok=True)
            if compute_vbench:
                vbench_dir = os.path.join(validation_path, "vbench")
                os.makedirs(vbench_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        total_samples = min(len(val_dataset), max_samples)

        # Pad with the last sample so each process receives the same number of items.
        samples = torch.arange(total_samples)
        if total_samples % accelerator.num_processes != 0:
            total_samples = math.ceil(total_samples / accelerator.num_processes) * accelerator.num_processes
            samples = torch.cat([samples, samples[-1].unsqueeze(0).repeat(total_samples - len(samples))])

        samples_per_process = total_samples // accelerator.num_processes
        start_idx = accelerator.process_index * samples_per_process
        end_idx = start_idx + samples_per_process

        logger.info(
            f"Process {accelerator.process_index} processing validation samples {start_idx} to {end_idx-1} "
            f"(total: {total_samples})",
            main_process_only=False
        )
        if "fid" in self.eval_metrics:
            fid_metric_cls = FIDDistMetric(normalize=True, device=accelerator.device)
        if "fvd" in self.eval_metrics:
            fvd_metric_cls = StyleGanFVDMetric(device=accelerator.device)

        processed_count = 0
        num_views = len(val_dataset.camera_names)
        for k in range(start_idx, end_idx):
            idx = samples[k % len(samples)].item() % len(val_dataset)
            val_dataset_kwargs["idx"] = idx
            val_dataset_kwargs["validation_mode"] = validation_mode
            if validation_mode.startswith("i2v") and "num_condition_images" in pipeline_kwargs:
                val_dataset_kwargs["num_condition_images"] = pipeline_kwargs["num_condition_images"]
            sample_data = val_dataset.__getitem__(val_dataset_kwargs)
            if sample_data is None:
                continue
            prompt = sample_data['text']
            logger.info(
                f"Process {accelerator.process_index} validating sample {idx+1}/{total_samples}. "
                f"Prompt: {prompt}",
                main_process_only=False,
            )
            model_mode = sample_data["model_mode"]
            clip_id = f"{k}_{sample_data['clip_id']}"
            metric_info = {}

            with torch.no_grad():
                with torch.autocast("cuda", dtype=weight_dtype):
                    num_frames, height, width = val_dataset_kwargs["resolution"]
                    video_length = int((num_frames - 1) // self.components.vae.config.temporal_compression_ratio * self.components.vae.config.temporal_compression_ratio) + 1 if num_frames != 1 else 1
                    for key, value in sample_data['conditions'].items():
                        sample_data['conditions'][key] = value.unsqueeze(0)

                    vis_additional_conditions = {}
                    for con_name, con_value in sample_data['conditions'].items():
                        if con_name == "bbox":
                            vis_additional_conditions[con_name] = (con_value[0].clone() + 1.0) / 2.0
                        if con_name == "bbox" and len(sample_data["ref_ttm_vis"]) != 0:
                            ref_img_vis = sample_data['ref_ttm_vis']
                            vis_additional_conditions["bbox"] = 0.5 * vis_additional_conditions["bbox"] + 0.5 * ref_img_vis

                    if model_mode == "i2v":
                        mask = sample_data['mask'].unsqueeze(0)     # b, f, 1, h, w
                        mask_video = sample_data['mask_pixel_values'].unsqueeze(0)
                        clip_image = sample_data['clip_pixel_values']   # b, h, w, c  ,[0, 255]
                    else:
                        mask_video, mask, clip_image = None, None, None

                    negative_prompt = WAN_FUN_NEGATIVE_PROMPT

                    if len(sample_data["ref_ttm_value"]) == 0:
                        sample_data['ref_ttm_value'] = None
                    if len(sample_data["ref_origin_ttm"]) == 0:
                        sample_data['ref_origin_ttm'] = None

                    pipe_kwargs = {
                        "prompt": prompt,
                        "video": sample_data["pixel_values"].unsqueeze(0),
                        "mask_video": mask_video,
                        "mask": mask,
                        "clip_image": clip_image,
                        "num_frames": video_length,
                        "negative_prompt": negative_prompt,
                        "height": height,
                        "width": width,
                        "guidance_scale": 6.0,
                        "generator": generator,
                        "num_cameras": num_views,
                        "additional_conditions": sample_data['conditions'],
                        "crossview_attn_type": self.args.crossview_attn_type,
                        "step": step,
                        "ttm": sample_data['ref_ttm_value'],
                        "ttm_origin": sample_data['ref_origin_ttm'],
                        **pipeline_kwargs,
                        **self.config.get("extra_pipeline_kwargs", {}),
                    }

                    videos = pipeline(**pipe_kwargs).videos
                    videos = rearrange(videos[0], "c f h w -> f c h w").contiguous()

                gt_videos = (sample_data['pixel_values'] + 1.0) / 2.0
                for eval_metric in self.eval_metrics:
                    if eval_metric == "met3r":
                        scores = evaluate_met3r(videos, num_views=num_views, debug=True)
                        metric_info["met3r_score"] = scores
                    if eval_metric == "fid" or eval_metric == "fvd":
                        if eval_metric == "fid":
                            fid_metric_cls.update(videos, real=False)
                            fid_metric_cls.update(gt_videos, real=True)
                        if eval_metric == "fvd":
                            raise NotImplementedError("FVD metric is not implemented yet.")
                            scores = fvd_metric_cls.compute(fake_videos, gt_videos)
                            metric_info["fvd_score"] = scores
                if len(metric_info) > 0:
                    metric_file = os.path.join(validation_path, f"{clip_id}_met3r.json")
                    if "met3r_score" in metric_info and "debug_info" in metric_info["met3r_score"]:
                        debug_info = metric_info["met3r_score"].pop("debug_info", None)
                    else:
                        debug_info = None
                    with open(metric_file, 'w') as f:
                        json.dump(metric_info, f, indent=4)

                filename = os.path.join(validation_path, f"{clip_id}.mp4")
                gt_filename = os.path.join(validation_path, f"{clip_id}_gt.mp4")
                comparison_filename = os.path.join(validation_path, f"{clip_id}_comparison.mp4")

                save_multiview_videos_grid(
                    videos,
                    comparison_filename,
                    fps=10,
                    gt_videos=gt_videos,
                    camera_names=val_dataset.camera_names,
                    met3r_metric=metric_info["met3r_score"]["met3r_view_scores"] if "met3r_score" in metric_info and "met3r_view_scores" in metric_info["met3r_score"] else None,
                    fid_metric=metric_info["fid_score"] if "fid_score" in metric_info else None,
                    fvd_metric=metric_info["fvd_score"] if "fvd_score" in metric_info else None,
                )

                save_multiview_videos_grid(
                    videos,
                    filename,
                    fps=10,
                    camera_names=val_dataset.camera_names,
                    met3r_metric=metric_info["met3r_score"]["met3r_view_scores"] if "met3r_score" in metric_info and "met3r_view_scores" in metric_info["met3r_score"] else None,
                    fid_metric=metric_info["fid_score"] if "fid_score" in metric_info else None,
                    fvd_metric=metric_info["fvd_score"] if "fvd_score" in metric_info else None,
                )
                save_multiview_videos_grid(gt_videos, gt_filename, fps=10, camera_names=val_dataset.camera_names)

                # Save condition visualizations next to the generated sample.
                if len(vis_additional_conditions) > 0:
                    for con_name, con_value in vis_additional_conditions.items():
                        overlap = con_name == "bbox"
                        save_multiview_videos_grid(
                            con_value,
                            os.path.join(validation_path, f"{clip_id}_{con_name}.mp4"),
                            gt_videos=None,
                            fps=10,
                            camera_names=val_dataset.camera_names,
                            overlap=overlap,
                        )

                if len(metric_info) > 0 and debug_info is not None:
                    debug_met3r_dir_name = os.path.join(validation_path, "debug", clip_id)
                    os.makedirs(debug_met3r_dir_name, exist_ok=True)
                    save_met3r_debug_images(debug_info, debug_met3r_dir_name)

                if compute_vbench:
                    compute_vbench_metric(
                        videos_path=filename,
                        name=clip_id,
                        output_path=vbench_dir,
                        device=accelerator.device,
                    )

            processed_count += 1

        if "fid" in self.eval_metrics and (end_idx - start_idx) > 0:
            fid = fid_metric_cls.compute()
            del fid_metric_cls
            if accelerator.is_main_process:
                all_fid_path = os.path.join(validation_path, "fid_scores.txt")
                with open(all_fid_path, 'w') as f:
                    f.write(f"All FID scores: {fid.item()}\n")

        accelerator.wait_for_everyone()

        if compute_vbench and accelerator.is_main_process:
            vbench_score = VbenchScore(
                results_dir=vbench_dir,
                model_name=model_mode,
            )
            vbench_score.calculate_all_scores_and_save_to_file(save_file=os.path.join(vbench_dir, "vbench_all_scores.json"))
            vbench_score.print_results()

        logger.info(
            f"Process {accelerator.process_index} completed validation. Processed {processed_count} samples.",
            main_process_only=False
        )

        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        if accelerator.is_main_process:
            logger.info(f"Validation completed successfully. Results saved to {validation_path}")
