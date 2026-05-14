import os
import math
import torch
import json
import numpy as np
from scipy.linalg import sqrtm
import torch.nn.functional as F
from met3r import MEt3R
from typing import Tuple, Optional
from torch import Tensor
from collections import defaultdict
from einops import rearrange
import matplotlib.pyplot as plt
import torch.distributed as dist
from torchmetrics.image.fid import NoTrainInceptionV3, _compute_fid
from composia.utils.constant import *


def evaluate_met3r(
    inputs,
    cam_pairs=[
        (1, 0),
        (0, 2),
        (2, 6),
        (6, 5),
        (5, 4),
        (4, 0),
    ],
    num_views=1,
    img_size=(256, 256),
    frame_interval=3,
    debug=False,
):
    """
    Evaluate MEt3R on the given inputs.

    Args:
        inputs (torch.Tensor): Input tensor of shape (batch, channels, views, height, width). value:(0, 1)
        img_size (int): Size of the input images.
        cam_pairs (list): List of camera pairs to evaluate.

    Returns:
        torch.Tensor: The computed MEt3R score.
    """
    inputs = rearrange(inputs, "b c (v f) h w -> b f v c h w", v=num_views)
    if frame_interval > 1:
        inputs = inputs[:, ::frame_interval, :, :, :, :]
    
    b, f = inputs.shape[:2]
    inputs = rearrange(inputs, "b f v c h w -> (b f v) c h w")
    inputs = (inputs - 0.5) * 2  # convert to [-1, 1] range
    if img_size is not None:
        inputs = F.interpolate(
            inputs,
            size=img_size,
            mode="bilinear",
            align_corners=False
        )
    inputs = rearrange(inputs, "(b f v) c h w -> (b f) v c h w", v=num_views, b=b, f=f)

    # Initialize MEt3R
    metric = MEt3R(
        img_size=img_size,
        use_norm=True,
        backbone="mast3r",
        feature_backbone="dino16",
        feature_backbone_weights="mhamilton723/FeatUp",
        upsampler="featup",
        distance="cosine",
        freeze=True,
    ).cuda()
    
    scores = dict()
    if debug:
        scores["debug_info"] = []

    for cam_pair in cam_pairs:
        cam_inputs = inputs[:, cam_pair, :, :, :].cuda()
        # Evaluate MEt3R
        score, *outputs = metric(
            images=cam_inputs,
            return_overlap_mask=debug,
            return_score_map=debug,
            return_projections=debug
        )
        if debug:
            # Store score, mask, score map, and projections for later visualization.
            for i in range(score.shape[0]):
                score_i = score[i].mean().item()
                debug_info = {
                    "flag": f"cam_{cam_pair[0]}__cam_{cam_pair[1]}__frame_{i}",
                    "score": score_i,
                    "image_1": cam_inputs[i, 0].cpu().numpy(),  # Image 1, shape (c, h, w)
                    "image_2": cam_inputs[i, 1].cpu().numpy(),  # Image 2, shape (c, h, w)
                    "overlap_mask": outputs[0][i].cpu().numpy(),  # Overlap mask, shape (height, width)
                    "score_map": outputs[1][i].cpu().numpy(),  # Score map, shape (height, width)
                    "projection": outputs[2][i].cpu().numpy()  # Projections, shape (num_views, height, width)
                }
                scores["debug_info"].append(debug_info)

        scores[f"{cam_pair[0]}__{cam_pair[1]}"] = score.mean().item()
        torch.cuda.empty_cache()
    
    inputs = rearrange(inputs, "(b f) v c h w -> (b v) f c h w", b=b, f=f, v=num_views)
    for image_id in range(f-1):
        cam_inputs = inputs[:, image_id:image_id+2, :, :, :].cuda()   # (b*v, 2, c, h, w)
        # cam_inputs = inputs[:2, image_id:image_id+2, :, :, :].cuda()   # (b*v, 2, c, h, w) test case
        # Evaluate MEt3R
        score, *outputs = metric(
            images=cam_inputs,
            return_overlap_mask=debug,
            return_score_map=debug,
            return_projections=debug
        )  # score shape: (b*v, 1)
        if debug:
            # Store score, mask, score map, and projections for later visualization.
            for i in range(score.shape[0]):
                score_i = score[i].mean().item()
                debug_info = {
                    "flag": f"cam_{i}__frame_{image_id}",
                    "image_id": image_id,
                    "score": score_i,
                    "image_1": (cam_inputs[i, 0].cpu().numpy() + 1.0) / 2.0,  # Image 1, shape (c, h, w)
                    "image_2": (cam_inputs[i, 1].cpu().numpy() + 1.0) / 2.0,  # Image 2, shape (c, h, w)
                    "overlap_mask": outputs[0][i].cpu().numpy(),  # Overlap mask, shape (height, width)
                    "score_map": outputs[1][i].cpu().numpy(),  # Score map, shape (height, width)
                    "projection": outputs[2][i].cpu().numpy()  # Projections, shape (num_views, height, width)
                }
                scores["debug_info"].append(debug_info)

        score = rearrange(score, "(b v) -> b v", b=b, v=num_views)
        # Average scores over the first dimension.
        score = score.mean(dim=0).cpu().numpy().tolist()  # shape: (v,)

        scores[f"frame_{image_id}"] = {}
        for cam_id, cam_score in enumerate(score):
            scores[f"frame_{image_id}"][cam_id] = cam_score
        torch.cuda.empty_cache()
    
    scores["met3r_view_scores"] = []
    for view_id in range(num_views):
        view_score = [scores[f"frame_{image_id}"][view_id] for image_id in range(f-1)]
        view_score = np.mean(view_score)
        scores["met3r_view_scores"].append(view_score)

    return scores


class StyleGanFVDMetric:
    pretrained_model_path=".cache/torch/hub/fvd/styleganv/i3d_torchscript.pt"
    def __init__(self, device='cpu'):
        self.device = device
        self.i3d = self.load_i3d_pretrained()
    
    def load_i3d_pretrained(self):
        i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
        if not os.path.exists(self.pretrained_model_path):
            print(f"preparing for download {i3D_WEIGHTS_URL}, you can download it by yourself.")
            os.system(f"wget {i3D_WEIGHTS_URL} -O {self.pretrained_model_path}")
        i3d = torch.jit.load(self.pretrained_model_path).eval().to(self.device).float()
        if self.device != 'cpu' and torch.cuda.device_count() > 1:
            i3d = torch.nn.DataParallel(i3d)
        return i3d
    
    def preprocess_single(self, video, resolution=224, sequence_length=None):
        # video: CTHW, [0, 1]
        video = video.float()
        c, t, h, w = video.shape

        # temporal crop
        if sequence_length is not None:
            assert sequence_length <= t
            video = video[:, :sequence_length]

        # scale shorter side to resolution
        scale = resolution / min(h, w)
        if h < w:
            target_size = (resolution, math.ceil(w * scale))
        else:
            target_size = (math.ceil(h * scale), resolution)
        video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

        # center crop
        c, t, h, w = video.shape
        w_start = (w - resolution) // 2
        h_start = (h - resolution) // 2
        video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

        # [0, 1] -> [-1, 1]
        video = (video - 0.5) * 2
        return video.contiguous()
    
    @torch.no_grad()
    def get_fvd_feats(self, videos):
        detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.
        feats = np.empty((0, 400)).astype(np.float32)  # Initialize an empty array to store features.
        x = torch.stack([self.preprocess_single(video) for video in videos]).to(self.device)
        feats = np.vstack([
            feats,
            self.i3d(x=x, **detector_kwargs).detach().float().cpu().numpy()
        ])
        return feats
    
    def frechet_distance(self, feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
        def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            mu = feats.mean(axis=0) # [d]
            sigma = np.cov(feats, rowvar=False) # [d, d]
            return mu, sigma
        mu_gen, sigma_gen = compute_stats(feats_fake)
        mu_real, sigma_real = compute_stats(feats_real)
        m = np.square(mu_gen - mu_real).sum()
        if feats_fake.shape[0]>1:
            s, _ = sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
            fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        else:
            fid = np.real(m)
        return float(fid)
    
    def compute(self, videos1, videos2, video_length=10):
        # videos [batch_size, timestamps, channel, h, w]
        assert videos1.shape == videos2.shape
        # BTCHW -> BCTHW
        videos1 = videos1.permute(0, 2, 1, 3, 4)
        videos2 = videos2.permute(0, 2, 1, 3, 4)
        
        # Compute FVD on the first, middle, and last `video_length` frame windows.
        start_idxs = [0, (videos1.shape[2] - video_length) // 2, videos1.shape[2] - video_length]
        end_idxs = [video_length, (videos1.shape[2] - video_length) // 2+ video_length, videos1.shape[2]]
        fvd_results = []
        for start_idx, end_idx in zip(start_idxs, end_idxs):
            # get video segments
            segment_videos1 = videos1[:, :, start_idx:end_idx, :, :]
            segment_videos2 = videos2[:, :, start_idx:end_idx, :, :]

            # get FVD features
            feats1 = self.get_fvd_feats(segment_videos1)
            feats2 = self.get_fvd_feats(segment_videos2)
            # calculate FVD
            fvd_results.append(self.frechet_distance(feats1, feats2))
        fvd_results = np.array(fvd_results)  # shape (b, 3)
        return fvd_results


class FIDMetric:
    def __init__(
        self,
        num_features: int = 2048,
        reset_real_features: bool = True,
        normalize: bool = False,
        feature_extractor_weights_path: Optional[str] = None,
        device='cpu',
        dtype=torch.float32,
    ) -> None:
        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize
        self.device = device
        self.dtype = dtype
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        valid_int_input = (64, 192, 768, 2048)
        assert num_features in valid_int_input, f"Integer input to argument `feature` must be one of {valid_int_input}"
        
        self.inception = NoTrainInceptionV3(
            name="inception-v3-compat",
            features_list=[str(num_features)],
            feature_extractor_weights_path=feature_extractor_weights_path,
        ).to(self.device)
        self.reset_real_features = reset_real_features
        self.num_features = num_features
        
        # Initialize metric states.
        mx_num_feats = (num_features, num_features)
        
        # Real-image statistics.
        self.real_features_sum = torch.zeros(num_features).double().to(device)
        self.real_features_cov_sum = torch.zeros(mx_num_feats).double().to(device)
        self.real_features_num_samples = torch.tensor(0).long().to(device)
        
        # Generated-image statistics.
        self.fake_features_sum = torch.zeros(num_features).double().to(device)
        self.fake_features_cov_sum = torch.zeros(mx_num_feats).double().to(device)
        self.fake_features_num_samples = torch.tensor(0).long().to(device)

    def update(self, imgs: Tensor, real: bool) -> None:
        """Update the state with extracted features."""
        imgs = imgs.detach().to(device=self.device, dtype=self.dtype)
        imgs = (imgs * 255).byte() if self.normalize else imgs
        features = self.inception(imgs)
        self.orig_dtype = features.dtype
        features = features.double()
        
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # Update the matching state bucket for real or generated images.
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += imgs.shape[0]
    
    def save_info(self, save_path):
        np.savez_compressed(
            save_path,
            real_features_sum=self.real_features_sum.cpu().numpy(),
            real_features_cov_sum=self.real_features_cov_sum.cpu().numpy(),
            real_features_num_samples=self.real_features_num_samples.cpu().numpy(),
            fake_features_sum=self.fake_features_sum.cpu().numpy(),
            fake_features_cov_sum=self.fake_features_cov_sum.cpu().numpy(),
            fake_features_num_samples=self.fake_features_num_samples.cpu().numpy(),
        )


def compute_fid_from_dir(result_dir) -> Tensor:
    real_features_sum = None
    real_features_cov_sum = None
    real_features_num_samples = None
    fake_features_sum = None
    fake_features_cov_sum = None
    fake_features_num_samples = None

    for result_name in os.listdir(result_dir):
        if result_name.endswith(".npz"):
            result_file = os.path.join(result_dir, result_name)
            state = np.load(result_file)
            if real_features_sum is None:
                real_features_sum = torch.tensor(state['real_features_sum']).double()
                real_features_cov_sum = torch.tensor(state['real_features_cov_sum']).double()
                real_features_num_samples = torch.tensor(state['real_features_num_samples']).long()
                fake_features_sum = torch.tensor(state['fake_features_sum']).double()
                fake_features_cov_sum = torch.tensor(state['fake_features_cov_sum']).double()
                fake_features_num_samples = torch.tensor(state['fake_features_num_samples']).long()
            else:
                real_features_sum += torch.tensor(state['real_features_sum']).double()
                real_features_cov_sum += torch.tensor(state['real_features_cov_sum']).double()
                real_features_num_samples += torch.tensor(state['real_features_num_samples']).long()
                fake_features_sum += torch.tensor(state['fake_features_sum']).double()
                fake_features_cov_sum += torch.tensor(state['fake_features_cov_sum']).double()
                fake_features_num_samples += torch.tensor(state['fake_features_num_samples']).long()
            
    if real_features_num_samples < 2 or fake_features_num_samples < 2:
        raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
    
    mean_real = (real_features_sum / real_features_num_samples).unsqueeze(0)
    mean_fake = (fake_features_sum / fake_features_num_samples).unsqueeze(0)
    cov_real_num = real_features_cov_sum - real_features_num_samples * mean_real.t().mm(mean_real)
    cov_real = cov_real_num / (real_features_num_samples - 1)
    cov_fake_num = fake_features_cov_sum - fake_features_num_samples * mean_fake.t().mm(mean_fake)
    cov_fake = cov_fake_num / (fake_features_num_samples - 1)
    return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(torch.float32)


class FIDDistMetric:
    def __init__(
        self,
        num_features: int = 2048,
        reset_real_features: bool = True,
        normalize: bool = False,
        feature_extractor_weights_path: Optional[str] = None,
        device='cpu',
        dtype=torch.float32,
    ) -> None:
        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize
        self.device = device
        self.dtype = dtype
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        valid_int_input = (64, 192, 768, 2048)
        assert num_features in valid_int_input, f"Integer input to argument `feature` must be one of {valid_int_input}"
        
        self.inception = NoTrainInceptionV3(
            name="inception-v3-compat",
            features_list=[str(num_features)],
            feature_extractor_weights_path=feature_extractor_weights_path,
        ).to(self.device)
        self.reset_real_features = reset_real_features
        self.num_features = num_features
        
        # Initialize metric states.
        mx_num_feats = (num_features, num_features)
        
        # Real-image statistics.
        self.real_features_sum = torch.zeros(num_features).double().to(device)
        self.real_features_cov_sum = torch.zeros(mx_num_feats).double().to(device)
        self.real_features_num_samples = torch.tensor(0).long().to(device)
        
        # Generated-image statistics.
        self.fake_features_sum = torch.zeros(num_features).double().to(device)
        self.fake_features_cov_sum = torch.zeros(mx_num_feats).double().to(device)
        self.fake_features_num_samples = torch.tensor(0).long().to(device)

    def _distributed_sync(self):
        """Run explicit distributed synchronization."""
        if self.is_distributed and dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            return
            
        # Synchronize real-image statistics.
        for tensor in [self.real_features_sum, self.real_features_cov_sum]:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.real_features_num_samples, op=dist.ReduceOp.SUM)
        
        # Synchronize generated-image statistics.
        for tensor in [self.fake_features_sum, self.fake_features_cov_sum]:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.fake_features_num_samples, op=dist.ReduceOp.SUM)
        
        # Prevent double synchronization.
        self.is_distributed = False

    def update(self, imgs: Tensor, real: bool) -> None:
        """Update the state with extracted features."""
        imgs = imgs.detach().to(device=self.device, dtype=self.dtype)
        imgs = (imgs * 255).byte() if self.normalize else imgs
        features = self.inception(imgs)
        self.orig_dtype = features.dtype
        features = features.double()
        
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # Update the matching state bucket for real or generated images.
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += imgs.shape[0]

    def compute(self) -> Tensor:
        """Calculate FID score with explicit synchronization before computation."""
        # Explicitly synchronize before computing the final metric.
        self._distributed_sync()
        
        # Ensure both distributions have enough samples.
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
        
        # Compute means and covariances.
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)
        
        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)
    
    def reset(self):
        """Reset all metric states."""
        # Reset the distributed flag.
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1
        
        # Reset real-image statistics.
        self.real_features_sum = torch.zeros(self.num_features).double().to(self.device)
        self.real_features_cov_sum = torch.zeros((self.num_features, self.num_features)).double().to(self.device)
        self.real_features_num_samples = torch.tensor(0).long().to(self.device)
        
        # Reset generated-image statistics.
        self.fake_features_sum = torch.zeros(self.num_features).double().to(self.device)
        self.fake_features_cov_sum = torch.zeros((self.num_features, self.num_features)).double().to(self.device)
        self.fake_features_num_samples = torch.tensor(0).long().to(self.device)


def compute_vbench_metric(
    videos_path,
    name,
    output_path,
    device="cpu",
    dimension=["subject_consistency", "background_consistency", "temporal_flickering", "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality"],  # , "human_action", "temporal_style", "overall_consistency"
):
    from vbench import VBench
    my_VBench = VBench(
        device = device,
        full_info_dir = None,
        output_path = output_path
    )

    kwargs = {}
    kwargs['imaging_quality_preprocessing_mode'] = "longer"
    print(f"Computing VBench metrics for videos in {videos_path} with name {name}...")
    my_VBench.evaluate(
        videos_path = videos_path,
        name = f'results_{name}',
        dimension_list = dimension,
        mode="custom_input",
        **kwargs
    )
    dist.barrier()

    # Aggregate scores from every `*_eval_results.json` file under the output directory.
    if dist.get_rank() == 0:
        for file in os.listdir(output_path):
            if file.endswith('_eval_results.json'):
                result_json = os.path.join(output_path, file)
                score_fn = VbenchScore(result_json)
                save_score_path = os.path.join(output_path, file.replace('_eval_results.json', '_all_scores.json'))
                score_fn.calculate_all_scores_and_save_to_file(save_score_path)


class VbenchScore:
    def __init__(self, result_json):
        self.result_json = result_json
        self.upload_data = {}
        self.normalized_score = {}
        self.quality_score = 0
        self.semantic_score = 0
        self.final_score = 0
    
    def load_results_from_json(self):
        """Load results directly from the evaluation_results directory."""
        if not os.path.exists(self.result_json):
            raise FileNotFoundError(f"Results directory '{self.result_json}' not found")
        
        try:
            with open(self.result_json, 'r') as f:
                result_json = json.load(f)
                if isinstance(result_json, dict):
                    for sub_key in result_json:
                        clean_key = sub_key.replace('_', ' ')
                        if isinstance(result_json[sub_key], list) and len(result_json[sub_key]) > 0:
                            self.upload_data[clean_key] = result_json[sub_key][0]
                        else:
                            self.upload_data[clean_key] = result_json[sub_key]
        except Exception as e:
            print(f"Error loading {self.result_json}: {e}")
        
        # Ensure all tasks exist, defaulting missing ones to zero.
        for key in QUALITY_LIST:
            if key not in self.upload_data:
                self.upload_data[key] = 0
        
        return self.upload_data
    
    def calculate_normalized_score(self):
        """Compute normalized scores."""
        for key in QUALITY_LIST:
            if key in self.upload_data:
                min_val = NORMALIZE_DIC[key]['Min']
                max_val = NORMALIZE_DIC[key]['Max']
                # Avoid divide-by-zero.
                if max_val - min_val == 0:
                    normalized_value = 0
                else:
                    normalized_value = (self.upload_data[key] - min_val) / (max_val - min_val)
                self.normalized_score[key] = normalized_value * DIM_WEIGHT[key]
            else:
                self.normalized_score[key] = 0
        
        return self.normalized_score
    
    def calculate_quality_score(self):
        """Compute the quality score."""
        quality_scores = []
        for key in QUALITY_LIST:
            if key in self.normalized_score:
                quality_scores.append(self.normalized_score[key])
        
        total_weight = sum([DIM_WEIGHT[i] for i in QUALITY_LIST if i in DIM_WEIGHT])
        if total_weight > 0:
            self.quality_score = sum(quality_scores) / total_weight
        else:
            self.quality_score = 0
        
        return self.quality_score
    
    def calculate_semantic_score(self):
        """Compute the semantic score."""
        semantic_scores = []
        for key in SEMANTIC_LIST:
            if key in self.normalized_score:
                semantic_scores.append(self.normalized_score[key])
        
        total_weight = sum([DIM_WEIGHT[i] for i in SEMANTIC_LIST if i in DIM_WEIGHT])
        if total_weight > 0:
            self.semantic_score = sum(semantic_scores) / total_weight
        else:
            self.semantic_score = 0
        
        return self.semantic_score
    
    def calculate_final_score(self):
        """Compute the final combined score."""
        quality_score = self.calculate_quality_score()
        semantic_score = self.calculate_semantic_score()
        
        if QUALITY_WEIGHT + SEMANTIC_WEIGHT > 0:
            self.final_score = (quality_score * QUALITY_WEIGHT + semantic_score * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)
        else:
            self.final_score = 0
        
        return self.final_score
    
    def calculate_all_scores_and_save_to_file(self, save_file):
        """Compute all scores."""
        self.load_results_from_json()
        self.calculate_normalized_score()
        self.calculate_quality_score()
        # self.calculate_semantic_score()
        # self.calculate_final_score()
        results = {
            'upload_data': self.upload_data,
            'normalized_score': self.normalized_score,
            'quality_score': self.quality_score,
            # 'semantic_score': self.semantic_score,
            # 'final_score': self.final_score
        }
        with open(save_file, 'w') as f:
            json.dump(results, f, indent=4)
    
    def print_results(self):
        """Print the aggregated results."""
        print(f"Results file: {self.result_json}")
        print(f"\nSubmission info: \n{self.upload_data} \n")
        
        print('+------------------|------------------+')
        print(f'|     quality score|{self.quality_score:.6f}|')
        # print(f'|    semantic score|{self.semantic_score:.6f}|')
        # print(f'|       total score|{self.final_score:.6f}|')
        print('+------------------|------------------+')


def aggregate_all_scores(result_dir):
    all_score_files = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if f.endswith('all_scores.json')]
    all_results = defaultdict(list)
    for score_file in all_score_files:
        try:
            with open(score_file, 'r') as f:
                result_json = json.load(f)
                if isinstance(result_json, dict):
                    for key in result_json:
                        if key in ['quality_score']:
                            all_results[key].append(result_json[key])
                        elif key == 'normalized_score':
                            for sub_key in result_json[key]:
                                all_results[sub_key].append(result_json[key][sub_key])
        except Exception as e:
            print(f"Error loading {score_file}: {e}")

    aggregated_results = {}
    for key in all_results:
        aggregated_results[key] = float(np.mean(all_results[key]))
    
    save_file = os.path.join(result_dir, 'all_scores_summary.json')
    with open(save_file, 'w') as f:
        json.dump(aggregated_results, f, indent=4)
    print(f"Aggregated results saved to {save_file}")


class MetricPlot:
    def __init__(self):
        self.data = {}
    
    def update(self, step, model_name, metrics: dict):
        if model_name not in self.data:
            self.data[model_name] = {
                'steps': defaultdict(list),
                'metrics': defaultdict(list)
            }
        for key, value in metrics.items():
            self.data[model_name]['steps'][key].append(step)
            self.data[model_name]['metrics'][key].append(value)

    def save_plots(self, model_name, save_dir):
        if not self.data or model_name not in self.data or not self.data[model_name]['metrics']:
            print("No data to plot.")
            return
        
        data = self.data[model_name]
        for key in sorted(data['metrics'].keys()):
            plt.figure()
            # Draw the metric curve.
            plt.plot(data["steps"][key], data["metrics"][key], label=key, marker='o')  # Add markers so data points are easier to read.
            
            # Annotate each point with its value.
            for x, y in zip(data["steps"][key], data["metrics"][key]):
                plt.text(x, y, f'{y:.4f}', fontsize=8, ha='right')
            
            # Set the y-axis range.
            if key in ['fid', 'FVD']:
                plt.ylim(0, 200)
            else:
                plt.ylim(0, 1.0)
            
            plt.xlabel('Step')
            plt.ylabel(key)
            plt.title(f'{key} over Steps')
            plt.legend()
            plt.grid(True)
            
            # Ensure the output directory exists.
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'{key}.png'))
            plt.close()
        
