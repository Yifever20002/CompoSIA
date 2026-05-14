import csv
import json
import os
import random
import time
from typing import Optional
from copy import deepcopy
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from func_timeout import FunctionTimedOut, func_timeout
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from collections import defaultdict
from videox_fun.utils.utils import import_cls
from videox_fun.data.dataset_image_video import (
    VIDEO_READER_TIMEOUT,
    MultiVideoReader_contextmanager,
    VideoReader_contextmanager,
    get_video_reader_batch,
    resize_frame,
)
from composia.datasets.base_dataset import (
    BaseDataset,
    get_normalized_intrinsics
)
from composia.utils.camera.pinhole import PinholeCamera
import cv2
from PIL import Image
import numpy as np
from composia.utils.bbox_utils import build_cuboid_bounding_box, cuboid3d_to_polyline, CLASS_COLORS

def generate_lane_move(velocity, T_output, lane_width=3.5, max_yaw_deg=5.0, direction='left'):
    if direction == 'right':
        actual_width = -lane_width
    else:
        actual_width = lane_width

    max_yaw_rad = np.radians(max_yaw_deg)
    # Use the absolute lane width when estimating the required physical distance.
    min_dist_needed = 1.875 * lane_width / np.tan(max_yaw_rad)
    
    needed_frames = int(np.ceil(min_dist_needed / (velocity + 1e-6)))
    total_frames = max(needed_frames, T_output)
    
    steps_array = np.arange(total_frames + 1)
    t_steps = steps_array / float(total_frames)
    
    global_x = steps_array * velocity
    global_y = actual_width * (10 * t_steps**3 - 15 * t_steps**4 + 6 * t_steps**5)
    
    dx = np.gradient(global_x)
    dy = np.gradient(global_y)
    dx[dx < 1e-6] = 1e-6 
    global_yaw = np.arctan2(dy, dx)
    
    drifts = []

    for i in range(T_output):
        theta_i, x_i, y_i = global_yaw[i], global_x[i], global_y[i]
        theta_next, x_next, y_next = global_yaw[i+1], global_x[i+1], global_y[i+1]
        
        c, s = np.cos(theta_i), np.sin(theta_i)
        mat_i = np.array([
            [c, -s, 0, x_i],
            [s,  c, 0, y_i],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])

        c_n, s_n = np.cos(theta_next), np.sin(theta_next)
        mat_next = np.array([
            [c_n, -s_n, 0, x_next],
            [s_n,  c_n, 0, y_next],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])
        
        mat_delta = np.linalg.inv(mat_i) @ mat_next
        drifts.append(mat_delta)
        
    return torch.tensor(np.array(drifts))


def load_local_ref(path):
    """
    Load a local RGBA image as an RGB tensor in [-1, 1] with shape (3, H, W).
    """
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGB", img.size, (0, 0, 0))
    bg.paste(img, mask=img.split()[3])
    img_tensor = TF.to_tensor(bg)       # [0, 1]
    img_tensor = img_tensor * 2 - 1     # [-1, 1]
    return img_tensor

def augment_ref_patch(
    patch: torch.Tensor,
    do_aug: bool = True,
    clip_scale_range=(0.9, 1.0),
    max_rotate: float = 10.0,
    scale_jitter=(0.9, 1.1),
    translate_frac=(0.05, 0.05),
    perspective_strength: float = 0.0,
    flip_prob: float = 0.5,
) -> torch.Tensor:
    """
    patch: [1, C, H, W]
    Return an augmented patch with the same spatial size.
    """
    if (not do_aug) or patch.ndim != 4:
        return patch

    B, C, H, W = patch.shape
    device = patch.device

    # Random horizontal flip only.
    if torch.rand(1, device=device).item() < flip_prob:
        patch = TF.hflip(patch)

    p = patch[0]  # [C, H, W]

    return p.unsqueeze(0)  # [1, C, H, W]


def valid_bbox_mask(bboxes, min_size=16, max_aspect_ratio=6.0, edit=False):
    # bboxes: (T, 4) -> [y1, y2, x1, x2]
    not_nan = ~np.isnan(bboxes).any(axis=1)

    if edit:
        return not_nan

    heights = bboxes[:, 1] - bboxes[:, 0]
    widths  = bboxes[:, 3] - bboxes[:, 2]

    size_ok = (heights >= min_size) & (widths >= min_size)

    min_hw = np.minimum(heights, widths)
    min_hw = np.where(min_hw <= 0, 1e-6, min_hw)

    aspect = np.maximum(heights, widths) / min_hw
    aspect_ok = aspect <= max_aspect_ratio

    valid = not_nan & size_ok & aspect_ok

    return valid


def nuscenes_select_top_objects(frames, EXCLUDE_LABELS, top_k=10):
    """
    Select the globally highest-ranked object ids from per-frame object order.
    If fewer than top_k objects remain after filtering EXCLUDE_LABELS, keep all.

    Args:
        frames: list of per-frame object dictionaries.
        EXCLUDE_LABELS: categories to exclude.
        top_k: number of object ids to select.

    Returns:
        selected_ids: set[int]
    """
    rank_sum = defaultdict(int)
    count = defaultdict(int)

    for objs in frames:
        obj_gt_bboxes_id = objs['gt_bboxes_id']
        obj_gt_labels_3d = objs['gt_labels_3d']
        for rank in range(len(obj_gt_bboxes_id)):
            if obj_gt_labels_3d[rank] in EXCLUDE_LABELS:
                continue
            oid = obj_gt_bboxes_id[rank]
            rank_sum[oid] += rank
            count[oid] += 1

    if not count:
        return set(), [[] for _ in frames]

    avg_rank = {oid: rank_sum[oid] / count[oid] for oid in count}

    if len(avg_rank) <= top_k:
        selected_ids = set(avg_rank.keys())
    else:
        selected_ids = set(sorted(avg_rank, key=avg_rank.get)[:top_k])

    return selected_ids

def nescenes_build_tracks_and_minmax_bboxes(
    obj_infos,
    pixel_values,
    EXCLUDE_LABELS,
    k_top=1000,
    fill_value=np.nan,
    clip_id=None,
    valid_mode=None,
    conditions=[],
    camera_names=[],
    delta_T=None,
    delta_T_origin=None,
    camera_intrinsics=None,
    cam2lidar=None,
    cam2vcs=None,
    lidar2vcs=None,
    valid_action_mode='default',
    valid_bbox_mode='default',
    valid_id_mode='default',
    ref_img_size=16,
):
    """
    Build object tracks and projected 2D boxes from full-video object metadata.

    Returns:
        - id_list: [k]
        - xyz_tracks: (k, T, 3)
        - bbox_tracks: (k, T, 4) -> (minh, maxh, minw, maxw)
    Missing or invalid projections are filled with fill_value.
    """

    def compute_transform_matrix(xyz, yaw):
        rotate_mat = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotate_mat
        transform_matrix[:3, 3] = xyz
        return transform_matrix


    T, C, H_new, W_new = pixel_values.shape

    # Select top objects by average rank while excluding ignored labels.
    selected_ids = nuscenes_select_top_objects(obj_infos, EXCLUDE_LABELS, k_top)
    id_list = sorted(list(selected_ids))
    k_eff = len(id_list)

    id2idx = {oid: idx for idx, oid in enumerate(id_list)}

    xyz_tracks = np.full((k_top, T, 3), fill_value, dtype=np.float32)
    minmax_tracks = np.full((k_top, T, 4), fill_value, dtype=np.float32)
    center_2D_tracks = np.full((k_top, T, 2), fill_value, dtype=np.float32)
    depth_tracks = np.full((k_top, T), fill_value, dtype=np.float32)

    bbox_projections_tensor = torch.zeros(
        (T, 3, H_new, W_new),
        dtype=pixel_values.dtype,
        device=pixel_values.device,
    )

    K = camera_intrinsics[:3, :3]
    H, W = 900, 1600
    scale_h = H_new / H
    scale_w = W_new / W
    current_drift = torch.eye(4, device=delta_T.device)
    drift_list = [current_drift]
    for t in range(T-1):
        current_drift = torch.linalg.inv(delta_T_origin[t]) @ drift_list[t] @ delta_T[t]
        drift_list.append(current_drift.clone())
    accumulated_drifts = torch.stack(drift_list).cpu().numpy()
    vcs2cam = np.linalg.inv(cam2vcs)
    vcs2lidar = cam2lidar @ vcs2cam
    cam2lidar = vcs2lidar @ accumulated_drifts @ cam2vcs

    camera_model = PinholeCamera(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        w=W,
        h=H,
        device="cpu",
    )

    obj_black_list = []
    obj_white_list = []
    id_white_list = []

    for t in range(T):
        objs = obj_infos[t]
        obj_gt_bboxes_3d = objs['gt_bboxes_3d']
        obj_gt_bboxes_id = objs['gt_bboxes_id']
        polylines = []
        for i_obj in range(len(obj_gt_bboxes_id)):
            obj = obj_gt_bboxes_3d[i_obj]
            oid = obj_gt_bboxes_id[i_obj]

            xyz = np.array([
                obj[0],
                obj[1],
                obj[2]
            ])
            yaw = - (obj[6])

            object_lwh = np.array([
                obj[3],
                obj[4],
                obj[5]
            ])

            # nuScenes coordinates: +x points right, +y points forward.
            if valid_action_mode == "vx0":
                if t == 0 and abs(xyz[1]) < 0.7 and xyz[0] < 0.0:
                    obj_black_list.append(oid)
            if valid_action_mode == "moveleft":
                if t == 0 and xyz[0] < 1.0 and xyz[1] < 0.0:
                    obj_black_list.append(oid)
            if valid_action_mode == "moveright":
                if t == 0 and xyz[0] > -1.0 and xyz[1] < 0.0:
                    obj_black_list.append(oid)
            if valid_bbox_mode == 'vx1':
                if t==0 and abs(xyz[0]) < 1.0 and xyz[1] > 0.0 and xyz[1] < 10.0:
                    print('car in front of me!')
                    obj_white_list.append(oid)
                if oid in obj_white_list:
                    xyz[1] = xyz[1] + t * 0.25
            if valid_bbox_mode == 'moveright':
                if t==0 and xyz[0] < -1.0 and xyz[1] > 0.0 and xyz[1] < 20.0:
                    print('left car cut in!')
                    obj_white_list.append(oid)
                if oid in obj_white_list:
                    phase = (np.pi / 2) * t / T
                    xyz[1] = xyz[1] + t * (-0.6)
                    xyz[0] = xyz[0] + 3.5 * np.sin(phase)
            if valid_bbox_mode == 'moveleft':
                if t==0 and xyz[0] < 5.0 and xyz[0] > 1.0 and xyz[1] > 0.0 and xyz[1] < 20.0:
                    print('right car cut in!')
                    obj_white_list.append(oid)
                if oid in obj_white_list:
                    phase = (np.pi / 2) * t / T
                    xyz[1] = xyz[1] + t * (-0.2)
                    xyz[0] = xyz[0] - 3.5 * np.sin(phase)
            if valid_id_mode != 'default':
                if abs(xyz[0]) < 1.0 and xyz[1] > 0.0:
                    if t == 0:
                        id_white_list.append(oid)

            object_to_world = compute_transform_matrix(xyz, yaw)

            cuboid_eight_vertices = build_cuboid_bounding_box(object_lwh[0], object_lwh[1], object_lwh[2], object_to_world)  # [8, 3]

            polyline = cuboid3d_to_polyline(cuboid_eight_vertices)

            if not oid in obj_black_list:
                polylines.append(polyline)
            if (valid_id_mode != 'default') and (oid not in id_white_list):
                continue
            
            # minmax_2D requires all projected corners to lie inside the image.
            # center_2D only requires the object center to be visible.
            minmax_2D, center_2D, depth = camera_model.get_2D_minmax(cam2lidar[t], cuboid_eight_vertices, clamp_to_border=False)
            if not np.isnan(minmax_2D).any():
                min_h, max_h, min_w, max_w = minmax_2D
                minmax_2D = np.array([
                    round(min_h * scale_h),
                    round(max_h * scale_h),
                    round(min_w * scale_w),
                    round(max_w * scale_w)
                ], dtype=np.float32)
            if not np.isnan(center_2D).any():
                c_h, c_w = center_2D
                center_2D = np.array([
                    round(c_h * scale_h),
                    round(c_w * scale_w)
                ], dtype=np.float32)
            
            j = id2idx[oid]
            xyz_tracks[j, t] = xyz
            minmax_tracks[j, t] = minmax_2D
            center_2D_tracks[j, t] = center_2D
            depth_tracks[j, t] = depth

        bbox_projection = camera_model.draw_line_depth(cam2lidar[t], polylines, radius=5, colors=np.array(CLASS_COLORS["Car"]))
        
        bp_np = bbox_projection[0]   # (H_old, W_old, 3)

        bp_np_resized = cv2.resize(
            bp_np,
            (W_new, H_new),
                interpolation=cv2.INTER_AREA,
        )

        if bp_np_resized.dtype == np.uint8:
            bp_np_resized = bp_np_resized.astype(np.float32) / 255.0
        else:
            bp_np_resized = bp_np_resized.astype(np.float32)

        bp_t = torch.from_numpy(bp_np_resized).permute(2, 0, 1)  # [1,3,H_new,W_new]

        bp_t = bp_t.to(device=pixel_values.device, dtype=pixel_values.dtype)

        bbox_projections_tensor[t] = bp_t.clamp_(0, 1)

    
    ref_render = torch.zeros_like(bbox_projections_tensor)   # [T,3,H_new,W_new]

    depth_render = torch.full(
        (T, 1, H_new, W_new),
        float("inf"),
        dtype=torch.float32,
        device=pixel_values.device,
    )

    depth_tracks_t = torch.from_numpy(depth_tracks).to(
        device=pixel_values.device,
        dtype=torch.float32,
    )

    if ("ref_img" in conditions or "ttm" in conditions):
        for j in range(k_eff):
            obj_id = id_list[j]
            minmax = minmax_tracks[j]      # (T, 4)
            center_2D = center_2D_tracks[j]
            valid = valid_bbox_mask(minmax, min_size=int(H_new/16))
            idxs = np.where(valid)[0]
            valid_2D = ~np.isnan(center_2D).any(axis=1)
            idxs_2D = np.where(valid_2D)[0]

            if idxs.size == 0:
                continue

            t_sel = int(np.random.choice(idxs))
            y1, y2, x1, x2 = minmax[t_sel].astype(int)

            y1 = max(0, min(y1, H_new - 1))
            y2 = max(y1 + 1, min(y2, H_new))
            x1 = max(0, min(x1, W_new - 1))
            x2 = max(x1 + 1, min(x2, W_new))

            frame = pixel_values[t_sel]                          # (C, H, W)
            patch = frame[:, y1:y2, x1:x2].clone().unsqueeze(0)          # (1, C, h, w)

            patch = augment_ref_patch(patch, do_aug=(valid_mode is None))

            if patch.shape[-1] < 1 or patch.shape[-2] < 1:
                continue

            # Convert from [-1, 1] to [0, 1] for visualization and blending.
            patch_vis = (patch * 0.5 + 0.5).clamp(0, 1)[0]   # (C=3, S, S)
            ph, pw = patch_vis.shape[-2], patch_vis.shape[-1]
            patch_vis = patch_vis.to(dtype=ref_render.dtype)

            valid_frames = np.intersect1d(idxs, idxs_2D)
            for t in valid_frames:
                z = depth_tracks_t[j, t]

                y1b, y2b, x1b, x2b = minmax[t].astype(int)

                y1c = max(0, min(y1b, H_new - 1))
                y2c = max(0, min(y2b, H_new))
                x1c = max(0, min(x1b, W_new - 1))
                x2c = max(0, min(x2b, W_new))

                if y2c <= y1c or x2c <= x1c:
                    continue

                target_h = y2c - y1c
                target_w = x2c - x1c

                patch_crop = F.interpolate(
                    patch_vis.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]

                if patch_crop.numel() == 0:
                    continue

                roi_depth = depth_render[t, :, y1c:y2c, x1c:x2c]   # (1,h',w')
                roi_new   = ref_render[t, :, y1c:y2c, x1c:x2c]     # (3,h',w')
                mask = (z < roi_depth)
                if not mask.any():
                    continue

                roi_depth[mask] = z

                mask3 = mask.expand_as(roi_new)              # (3,h',w')
                roi_new[mask3] = patch_crop[mask3]

    xyz_tracks = torch.from_numpy(xyz_tracks).to(device=pixel_values.device, dtype=torch.float32)
    minmax_tracks = torch.from_numpy(minmax_tracks).to(device=pixel_values.device, dtype=torch.float32)
    center_2D_tracks = torch.from_numpy(center_2D_tracks).to(device=pixel_values.device, dtype=torch.float32)

    return_item = {
        "xyz": xyz_tracks,                    # [k, T, 3]
        "minmax_2D": minmax_tracks,           # [k, T, 4]
        "center_2D": center_2D_tracks,        # [k, T, 2]
        "bbox_render": bbox_projections_tensor,   #[T, 3, H_new, W_new]
        "ref_render": ref_render,
        "ap_bbox_render": 0.5 * bbox_projections_tensor + 0.5 * ref_render   #[T, 3, H_new, W_new]
    }

    return return_item


def nescenes_build_tracks_and_minmax_bboxes_apedit(
    obj_infos,
    pixel_values,
    EXCLUDE_LABELS,
    k_top=1000,
    fill_value=np.nan,
    clip_id=None,
    valid_mode=None,
    conditions=[],
    camera_names=[],
    delta_T=None,
    delta_T_origin=None,
    camera_intrinsics=None,
    cam2lidar=None,
    cam2vcs=None,
    lidar2vcs=None,
    valid_action_mode='default',
    valid_bbox_mode='default',
    valid_id_mode='default',
    ref_img_size=16,
):
    """
    Build object tracks and projected 2D boxes from full-video object metadata.

    Returns:
        - id_list: [k]
        - xyz_tracks: (k, T, 3)
        - bbox_tracks: (k, T, 4) -> (minh, maxh, minw, maxw)
    Missing or invalid projections are filled with fill_value.
    """

    def compute_transform_matrix(xyz, yaw):
        rotate_mat = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotate_mat
        transform_matrix[:3, 3] = xyz
        return transform_matrix


    T, C, H_new, W_new = pixel_values.shape

    # Select top objects by average rank while excluding ignored labels.
    selected_ids = nuscenes_select_top_objects(obj_infos, EXCLUDE_LABELS, k_top)
    id_list = sorted(list(selected_ids))
    k_eff = len(id_list)

    id2idx = {oid: idx for idx, oid in enumerate(id_list)}

    xyz_tracks = np.full((k_top, T, 3), fill_value, dtype=np.float32)
    minmax_tracks = np.full((k_top, T, 4), fill_value, dtype=np.float32)
    center_2D_tracks = np.full((k_top, T, 2), fill_value, dtype=np.float32)
    depth_tracks = np.full((k_top, T), fill_value, dtype=np.float32)

    bbox_projections_tensor = torch.zeros(
        (T, 3, H_new, W_new),
        dtype=pixel_values.dtype,
        device=pixel_values.device,
    )

    K = camera_intrinsics[:3, :3]
    H, W = 900, 1600
    scale_h = H_new / H
    scale_w = W_new / W
    current_drift = torch.eye(4, device=delta_T.device)
    drift_list = [current_drift]
    for t in range(T-1):
        current_drift = torch.linalg.inv(delta_T_origin[t]) @ drift_list[t] @ delta_T[t]
        drift_list.append(current_drift.clone())
    accumulated_drifts = torch.stack(drift_list).cpu().numpy()
    vcs2cam = np.linalg.inv(cam2vcs)
    vcs2lidar = cam2lidar @ vcs2cam
    cam2lidar = vcs2lidar @ accumulated_drifts @ cam2vcs

    camera_model = PinholeCamera(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        w=W,
        h=H,
        device="cpu",
    )

    ref_id, ref_l, ref_w, ref_h = valid_id_mode.split('-')
    ref_path = f"./assets/car_ref_imgs_rgba/{ref_id}.png"

    obj_black_list = []
    obj_white_list = []
    id_white_list = []

    for t in range(T):
        objs = obj_infos[t]
        obj_gt_bboxes_3d = objs['gt_bboxes_3d']
        obj_gt_bboxes_id = objs['gt_bboxes_id']
        polylines = []
        for i_obj in range(len(obj_gt_bboxes_id)):
            obj = obj_gt_bboxes_3d[i_obj]
            oid = obj_gt_bboxes_id[i_obj]

            xyz = np.array([
                obj[0],
                obj[1],
                obj[2]
            ])
            yaw = - (obj[6])

            object_lwh = np.array([
                obj[3],
                obj[4],
                obj[3]
            ])

            # nuScenes coordinates: +x points right, +y points forward.
            if valid_action_mode == "vx0":
                if t == 0 and abs(xyz[1]) < 0.7 and xyz[0] < 0.0:
                    obj_black_list.append(oid)
            if valid_action_mode == "moveleft":
                if t == 0 and xyz[0] < 1.0 and xyz[1] < 0.0:
                    obj_black_list.append(oid)
            if valid_action_mode == "moveright":
                if t == 0 and xyz[0] > -1.0 and xyz[1] < 0.0:
                    obj_black_list.append(oid)
            if valid_bbox_mode == 'vx1':
                if t==0 and abs(xyz[0]) < 1.0 and xyz[1] > 0.0 and xyz[1] < 10.0:
                    print('car in front of me!')
                    obj_white_list.append(oid)
                if oid in obj_white_list:
                    xyz[1] = xyz[1] + t * 0.25
            if valid_bbox_mode == 'moveright':
                if t==0 and xyz[0] < -1.0 and xyz[1] > 0.0 and xyz[1] < 20.0:
                    print('left car cut in!')
                    obj_white_list.append(oid)
                if oid in obj_white_list:
                    phase = (np.pi / 2) * t / T
                    xyz[1] = xyz[1] + t * (-0.0)
                    xyz[0] = xyz[0] + 3.5 * np.sin(phase)
            if valid_bbox_mode == 'moveleft':
                if t==0 and xyz[0] < 5.0 and xyz[0] > 1.0 and xyz[1] > 0.0 and xyz[1] < 20.0:
                    print('right car cut in!')
                    obj_white_list.append(oid)
                if oid in obj_white_list:
                    phase = (np.pi / 2) * t / T
                    xyz[1] = xyz[1] + t * (-0.0)
                    xyz[0] = xyz[0] - 3.5 * np.sin(phase)
            if abs(xyz[0]) < 1.0 and xyz[1] > 0.0:
                if t == 0:
                    id_white_list.append(oid)
                old_object_lwh = object_lwh
                object_lwh = np.array([float(ref_w), float(ref_l), float(ref_h)])
                old_z = xyz[2]
                xyz[2] = old_z - (old_object_lwh[2] / 2.0) + (object_lwh[2] / 2.0)

            object_to_world = compute_transform_matrix(xyz, yaw)

            cuboid_eight_vertices = build_cuboid_bounding_box(object_lwh[0], object_lwh[1], object_lwh[2], object_to_world)  # [8, 3]

            polyline = cuboid3d_to_polyline(cuboid_eight_vertices)

            if not oid in obj_black_list:
                polylines.append(polyline)
            if oid not in id_white_list:
                continue
            # minmax_2D requires all projected corners to lie inside the image.
            # center_2D only requires the object center to be visible.
            minmax_2D, center_2D, depth = camera_model.get_2D_minmax(cam2lidar[t], cuboid_eight_vertices, clamp_to_border=False)
            if not np.isnan(minmax_2D).any():
                min_h, max_h, min_w, max_w = minmax_2D
                minmax_2D = np.array([
                    round(min_h * scale_h),
                    round(max_h * scale_h),
                    round(min_w * scale_w),
                    round(max_w * scale_w)
                ], dtype=np.float32)
            if not np.isnan(center_2D).any():
                c_h, c_w = center_2D
                center_2D = np.array([
                    round(c_h * scale_h),
                    round(c_w * scale_w)
                ], dtype=np.float32)
            
            j = id2idx[oid]
            xyz_tracks[j, t] = xyz
            minmax_tracks[j, t] = minmax_2D
            center_2D_tracks[j, t] = center_2D
            depth_tracks[j, t] = depth

        bbox_projection = camera_model.draw_line_depth(cam2lidar[t], polylines, radius=5, colors=np.array(CLASS_COLORS["Car"]))

        bp_np = bbox_projection[0]   # (H_old, W_old, 3)

        bp_np_resized = cv2.resize(
            bp_np,
            (W_new, H_new),
                interpolation=cv2.INTER_AREA,
        )

        if bp_np_resized.dtype == np.uint8:
            bp_np_resized = bp_np_resized.astype(np.float32) / 255.0
        else:
            bp_np_resized = bp_np_resized.astype(np.float32)

        bp_t = torch.from_numpy(bp_np_resized).permute(2, 0, 1)  # [1,3,H_new,W_new]

        bp_t = bp_t.to(device=pixel_values.device, dtype=pixel_values.dtype)

        bbox_projections_tensor[t] = bp_t.clamp_(0, 1)

    ref_img = None
    
    ref_render = torch.zeros_like(bbox_projections_tensor)   # [T,3,H_new,W_new]

    depth_render = torch.full(
        (T, 1, H_new, W_new),
        float("inf"),
        dtype=torch.float32,
        device=pixel_values.device,
    )

    depth_tracks_t = torch.from_numpy(depth_tracks).to(
        device=pixel_values.device,
        dtype=torch.float32,
    )

    if ("ref_img" in conditions or "ttm" in conditions):
        for j in range(k_eff):
            obj_id = id_list[j]
            minmax = minmax_tracks[j]      # (T, 4)
            center_2D = center_2D_tracks[j]
            valid = valid_bbox_mask(minmax, min_size=int(H_new/16), edit=True)
            idxs = np.where(valid)[0]
            valid_2D = ~np.isnan(center_2D).any(axis=1)
            idxs_2D = np.where(valid_2D)[0]

            if idxs.size == 0:
                continue

            t_sel = int(np.random.choice(idxs))
            y1, y2, x1, x2 = minmax[t_sel].astype(int)

            y1 = max(0, min(y1, H_new - 1))
            y2 = max(y1 + 1, min(y2, H_new))
            x1 = max(0, min(x1, W_new - 1))
            x2 = max(x1 + 1, min(x2, W_new))

            ref_img = load_local_ref(ref_path).to(device=ref_render.device, dtype=ref_render.dtype)
            patch = ref_img.unsqueeze(0)

            if patch.shape[-1] < 1 or patch.shape[-2] < 1:
                continue

            # Convert from [-1, 1] to [0, 1] for visualization and blending.
            patch_vis = (patch * 0.5 + 0.5).clamp(0, 1)[0]   # (C=3, S, S)
            ph, pw = patch_vis.shape[-2], patch_vis.shape[-1]
            patch_vis = patch_vis.to(dtype=ref_render.dtype)

            valid_frames = np.intersect1d(idxs, idxs_2D)
            for t in valid_frames:
                z = depth_tracks_t[j, t]

                y1b, y2b, x1b, x2b = minmax[t].astype(int)

                y1c = max(0, min(y1b, H_new - 1))
                y2c = max(0, min(y2b, H_new))
                x1c = max(0, min(x1b, W_new - 1))
                x2c = max(0, min(x2b, W_new))

                if y2c <= y1c or x2c <= x1c:
                    continue

                target_h = y2c - y1c
                target_w = x2c - x1c

                patch_crop = F.interpolate(
                    patch_vis.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]

                if patch_crop.numel() == 0:
                    continue

                roi_depth = depth_render[t, :, y1c:y2c, x1c:x2c]   # (1,h',w')
                roi_new   = ref_render[t, :, y1c:y2c, x1c:x2c]     # (3,h',w')
                mask = (z < roi_depth)
                if not mask.any():
                    continue

                roi_depth[mask] = z

                mask3 = mask.expand_as(roi_new)              # (3,h',w')
                roi_new[mask3] = patch_crop[mask3]

    xyz_tracks = torch.from_numpy(xyz_tracks).to(device=pixel_values.device, dtype=torch.float32)
    minmax_tracks = torch.from_numpy(minmax_tracks).to(device=pixel_values.device, dtype=torch.float32)
    center_2D_tracks = torch.from_numpy(center_2D_tracks).to(device=pixel_values.device, dtype=torch.float32)

    return_item = {
        "xyz": xyz_tracks,                    # [k, T, 3]
        "minmax_2D": minmax_tracks,           # [k, T, 4]
        "center_2D": center_2D_tracks,        # [k, T, 2]
        "bbox_render": bbox_projections_tensor,   #[T, 3, H_new, W_new]
        "ref_render": ref_render,
        "ap_bbox_render": 0.5 * bbox_projections_tensor + 0.5 * ref_render   #[T, 3, H_new, W_new]
    }

    return return_item
