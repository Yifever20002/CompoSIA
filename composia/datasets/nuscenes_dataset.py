import csv
import os
import random
import numpy as np
from typing import Any, Dict, Literal, List, Optional
import torch
import torchvision.transforms as transforms
import pickle
import logging
from pyquaternion import Quaternion
from composia.datasets.base_dataset import (
    BaseDataset,
    get_normalized_intrinsics
)
from composia.utils.camera.pinhole import PinholeCamera
from composia.utils.editing_utils import (
    generate_lane_move,
    nescenes_build_tracks_and_minmax_bboxes,
    nescenes_build_tracks_and_minmax_bboxes_apedit,
)
from PIL import Image

import traceback

def obtain_next2top(first, current, epsilon=1e-6, v2=True):
    l2e_r = first["lidar2ego_rotation"]
    l2e_t = first["lidar2ego_translation"]
    e2g_r = first["ego2global_rotation"]
    e2g_t = first["ego2global_translation"]
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    l2e_r_s = current["lidar2ego_rotation"]
    l2e_t_s = current["lidar2ego_translation"]
    e2g_r_s = current["ego2global_rotation"]
    e2g_t_s = current["ego2global_translation"]

    # Build the transform from the current sweep to the first top LiDAR frame.
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    next2lidar_rotation = R.T  # points @ R.T + T
    next2lidar_translation = T
    if v2:
        # Invert the transform so points move from the anchor LiDAR to the next frame.
        _R = np.concatenate([next2lidar_rotation.T, np.array(
            [[0.,] * 3], dtype=T.dtype)], axis=0)
        _T = -next2lidar_rotation.T @ next2lidar_translation
        _T = np.concatenate(
            [_T[..., np.newaxis], np.array([[1.]], dtype=T.dtype)], axis=0)
        next2lidar = np.concatenate([_R, _T], axis=1)
    else:
        _R = np.concatenate(
            [next2lidar_rotation, np.array([[0.,]] * 3, dtype=T.dtype)], axis=1)
        _T = np.concatenate(
            [next2lidar_translation, np.array([1.], dtype=T.dtype)], axis=0)
        next2lidar = np.concatenate(
            [_R, _T[np.newaxis, ...]], axis=0,
        ).T
    if epsilon is not None:
        next2lidar[np.abs(next2lidar) < epsilon] = 0.
    return next2lidar



class nuScenesDataset(BaseDataset):
    """
    nuScenes video dataset used by the open-source eval path.
    """

    def __init__(
        self,
        ann_path: str,
        mode="train",
        video_repeat=0,
        ref_camera=None,
        camera_names=["camera_front"],
        video_sample_stride: int = 1,
        video_length_drop_start: float = 0.0,
        video_length_drop_end: float = 1.0,
        text_drop_ratio: float = 0.1,
        i2v_random_mask_probs: Dict[Literal["first_image", "random_middle_image", "random_first_n_images", "drop_last"], float] = {"first_image": 1.0},
        valid_conditions: list = [],
        conditions_kwargs: Dict[str, Any] = {},
        bbox_use_ap = False,
        samples_path = None,
        video_length = 17,
    ):
        
        self.samples_path = samples_path
        self.ann_file = ann_path
        self.video_sample_stride = video_sample_stride
        self.balance_keywords = None
        self.mode = mode
        self.video_length = video_length
        if self.mode == 'train':
            self.start_on_firstframe = False
            self.start_on_keyframe = True
        else:
            self.start_on_firstframe = True
            self.start_on_keyframe = False
        
        self.data_infos = self.load_annotations(self.ann_file)
        
        if len(self.clip_infos) == 0:
            raise ValueError("No video data found in annotation file")
        self.ref_camera = ref_camera
        self.camera_names = ['CAM_FRONT']
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end
        self.text_drop_ratio = text_drop_ratio
        self.i2v_random_mask_probs = i2v_random_mask_probs
        self.length = len(self.clip_infos)
        
        self.valid_conditions = valid_conditions
        self.conditions_kwargs = conditions_kwargs
        self.use_valid_flag = True
        self.CLASSES = [
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
        ]

    def build_clips(self, data_infos, scene_tokens):
        """Since the order in self.data_infos may change on loading, we
        calculate the index for clips after loading.

        Args:
            data_infos (list of dict): loaded data_infos
            scene_tokens (2-dim list of str): 2-dim list for tokens to each
            scene 

        Returns:
            2-dim list of int: int is the index in self.data_infos
        """
        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)}
        if self.balance_keywords is not None:
            data_infos, scene_tokens = self.balance_annotations(
                data_infos, scene_tokens)
        all_clips = []
        skip1, skip2 = 0, 0
        for sid, scene in enumerate(scene_tokens):
            if self.video_length == "full":
                clip = [self.token_data_dict[token] for token in scene]
                if self.micro_frame_size is not None:
                    res = len(clip) % self.micro_frame_size - 1
                    if res > 0:
                        clip = clip[:-res]
                all_clips.append(clip)
            else:
                assert isinstance(self.video_length, int)
                if sid in []:
                    logging.info(f"Got {len(all_clips)} for sid={sid}.")
                if self.start_on_firstframe:
                    first_frames = [0]
                else:
                    first_frames = range(len(scene) - self.video_length + 1)
                for start in first_frames:
                    if self.start_on_keyframe and ";" in scene[start]:
                        skip1 += 1
                        continue
                    if self.start_on_keyframe and len(scene[start]) >= 33:
                        skip2 += 1
                        continue
                    clip = [self.token_data_dict[token]
                            for token in scene[start: start + self.video_length]]
                    all_clips.append(clip)
        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Cut into {self.video_length}-clip, "
                     f"which has {len(all_clips)} in total. We skip {skip1} + "
                     f"{skip2} = {skip1 + skip2} possible starting frames. "
                     f"start_on_firstframe={self.start_on_firstframe}")
        return all_clips
    
    def load_annotations(self, ann_file: str) -> list:
        """
        Read the annotation file and return the ordered frame metadata.
        
        Args:
            ann_path (str): Path to the annotation file pickle.
        
        Returns:
            list: List of video data dictionaries
        """
        
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.video_sample_stride]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        self.clip_infos = self.build_clips(data_infos, data['scene_tokens'])

        return data_infos
    
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_bboxes_id = np.array(info["gt_box_ids"])[mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # Keep the raw box tensor; no center convention conversion is applied here.
        gt_bboxes_3d = torch.as_tensor(gt_bboxes_3d, dtype=torch.float32)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_bboxes_id=gt_bboxes_id,
        )
        return anns_results, mask

    def get_data_info_single(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]
        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            location=info["location"],
        )
        add_key = [
            "description",
            "timeofday",
            "visibility",
            "flip_gt",
        ]
        for key in add_key:
            if key in info:
                data[key] = info[key]

        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        data["image_paths"] = []
        data["lidar2camera"] = []
        data["lidar2image"] = []
        data["camera2ego"] = []
        data["camera_intrinsics"] = []
        data["camera2lidar"] = []
        data["ego2global"] = []

        for _, camera_info in info["cams"].items():
            data["image_paths"].append(camera_info["data_path"])

            lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            data["lidar2camera"].append(lidar2camera_rt.T)

            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
            data["camera_intrinsics"].append(camera_intrinsics)

            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            data["lidar2image"].append(lidar2image)

            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                camera_info["sensor2ego_rotation"]
            ).rotation_matrix
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
            data["camera2ego"].append(camera2ego)

            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            data["camera2lidar"].append(camera2lidar)

            ego2global = np.eye(4).astype(np.float32)
            ego2global[:3, :3] = Quaternion(
                camera_info["ego2global_rotation"]
            ).rotation_matrix
            ego2global[:3, 3] = camera_info["ego2global_translation"]
            data["ego2global"].append(ego2global)

        annos, mask = self.get_ann_info(index)
        if "visibility" in data:
            data["visibility"] = data["visibility"][mask]
        data["ann_info"] = annos
        return data

    def load_clip(self, clip):
        frames = []
        first_info = self.data_infos[clip[0]]
        for frame in clip:
            frame_info = self.get_data_info_single(frame)
            info = self.data_infos[frame]
            next2top = obtain_next2top(first_info, info)
            frame_info['next2top'] = next2top
            frames.append(frame_info)
        return frames

    def get_data_info(self, index):
        """We should sample from clip_infos
        """
        clip = self.clip_infos[index]
        frames = self.load_clip(clip)
        return frames

    def get_all_cam_video_paths(self, path: str, camera_names, condition_postfix=None) -> list:
        all_cam_video_paths = []
        for camera_name in camera_names:
            if condition_postfix is None:
                cam_path = os.path.join(path, f"{camera_name}_undistort.mp4")
            else:
                cam_path = os.path.join(path, f"{camera_name}_undistort_{condition_postfix}.mp4")
            if not os.path.exists(cam_path):
                return None
            all_cam_video_paths.append(cam_path)
        return all_cam_video_paths
    
    def read_and_process_images(
        self,
        image_paths: List[str],
        target_height: int,
        target_width: int,
        image_transforms: Optional[transforms.Compose] = None,
    ) -> torch.Tensor:
        """
        Read and process a list of image frames.
        
        Returns:
            torch.Tensor: Processed tensor [F, C, H, W] (F=Frames/List Length)
        """
        
        if image_transforms is None:
            image_transforms = transforms.Compose([
                transforms.Resize((target_height, target_width)), 
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])

        processed_frames = []

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')

                img = img.resize((target_width, target_height)) 
                img_tensor = transforms.functional.to_tensor(img)
                
                processed_frames.append(img_tensor)
                
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")
                raise ValueError(f"Failed to load image: {img_path}")

        if not processed_frames:
            raise ValueError("No frames were loaded.")
            
        pixel_values = torch.stack(processed_frames)

        pixel_values = image_transforms(pixel_values)
        
        return pixel_values


    def get_sample_index(
        self,
        mode,
        video_sample_n_frames,
        video_reader_length,
    ):
        min_sample_n_frames = min(
            video_sample_n_frames,
            int(video_reader_length * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
        )
        if min_sample_n_frames < video_sample_n_frames or (min_sample_n_frames - 1) % 4 != 0:
            raise ValueError(f"Video has insufficient frames for sampling. ")
        
        video_length = int(self.video_length_drop_end * video_reader_length)
        clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
        if mode == 'train':
            start_idx = random.randint(
                int(self.video_length_drop_start * video_length),
                video_length - clip_length
            ) if video_length != clip_length else 0
        else:
            start_idx = 0
        
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)
        return batch_index

    def load_whitelist(self, txt_path):
            """Read a simple whitelist text file with optional `#` comments."""
            valid_list = []
            if not os.path.exists(txt_path):
                return []
                
            with open(txt_path, 'r') as f:
                for line in f:
                    content = line.split('#')[0].strip()
                    if content:
                        valid_list.append(content)
            return valid_list

    def get_batch(
        self,
        idx: int,
        resoultion: list = [17, 240, 480],
        conditions: list = [],
        valid_mode = None,
    ):
        """
        Load and process a video batch
        
        Args:
            idx (int): Index of the video to load
            
        Returns:
            tuple: (pixel_values, text, data_type) where pixel_values is torch.Tensor [F, C, H, W]
        """

        video_sample_n_frames, train_height, train_width = resoultion

        frames = self.get_data_info(idx)
        assert video_sample_n_frames == len(frames), f"mismatch video length between {video_sample_n_frames} and {len(frames)}"

        camera2ego = frames[0]['camera2ego'][0]
        lidar2ego = frames[0]['lidar2ego']
        camera2lidar = [frame['camera2lidar'][0] for frame in frames]
        ego2global = [frame['ego2global'][0] for frame in frames]

        camera_intrinsics = frames[0]['camera_intrinsics'][0]

        old_prefix = '../data/nuscenes'
        all_cam_images_paths = []
        for frame in frames:
            for path in frame['image_paths']:
                if self.camera_names[0] in path:
                    new_path = path.replace(old_prefix, self.samples_path)
                    all_cam_images_paths.append(new_path)
                    break
        
        tokens = [frame['token'] for frame in frames]
        clip_id = tokens[0]

        valid_action_mode = "default"
        valid_bbox_mode = "default"
        valid_id_mode = "default"
        filter_mode = "default"

        if valid_mode is not None:
            try:
                # Validation mode names follow the YAML convention documented in config.
                _, valid_action_mode, valid_bbox_mode, valid_id_mode, filter_mode, _ = valid_mode.split('_', 5)
                print(f"ego action: {valid_action_mode}")
                print(f"BBox: {valid_bbox_mode}")
                print(f"ttm edit: {valid_id_mode}")
                print(f"filter mode: {filter_mode}")
            except ValueError:
                print("invalid validation mode format")

        if filter_mode != "default":
            text_path = f"assets/eval_select_nuScenes/{filter_mode}.txt"
            whitelist = self.load_whitelist(text_path)
            if not any(clip_id in item for item in whitelist):
                assert False

        bbox_pixel_values = None
        bbox_ap_conds = None
        ref_ttm_value = None
        ref_origin_ttm = None
        ref_ttm_vis = None

        pixel_values = self.read_and_process_images(
            all_cam_images_paths,
            train_height,
            train_width,
        )
        delta_T = torch.eye(4, device=pixel_values.device, dtype=pixel_values.dtype).unsqueeze(0).repeat(pixel_values.shape[0], 1, 1)
        delta_T_origin = delta_T.clone()

        if "action" in conditions:
            vcs2worlds = ego2global
            init_world2vcs = np.linalg.inv(vcs2worlds[0])
            F = video_sample_n_frames
            vcs2world_list = []
            actions = []
            delta_T = []

            for i in range(F):
                vcs2world = init_world2vcs @ vcs2worlds[i]
                vcs2world_list.append(vcs2world)
 
            for i in range(F - 1):
                T_i = vcs2world_list[i]
                T_next = vcs2world_list[i + 1]
                T_rel = np.linalg.inv(T_i) @ T_next
                delta = T_rel[:3, 3]
                actions.append(torch.tensor(delta, dtype=torch.float32))
                delta_T.append(torch.tensor(T_rel, dtype=torch.float32))

            actions = torch.stack(actions, dim=0)
            last_action = actions[-1:] 
            actions = torch.cat([actions, last_action], dim=0)[:, :2]

            delta_T = torch.stack(delta_T, dim=0)
            last_delta_T = delta_T[-1:]
            delta_T = torch.cat([delta_T, last_delta_T], dim=0)

            actions_origin = actions.clone()
            delta_actions = torch.zeros_like(actions)
            delta_T_origin = delta_T.clone()

            vcs2world_save = torch.stack([torch.from_numpy(m) for m in vcs2world_list], dim=0)

            if valid_action_mode != "default":
                if valid_action_mode == 'vx0':
                    actions[..., 0] = 0.
                    actions[..., 1] = 0.
                    delta_T = torch.eye(4, device=actions.device, dtype=actions.dtype).unsqueeze(0).repeat(delta_T_origin.shape[0], 1, 1)
                    delta_T[:, 0, 3] = actions[:, 0]
                    delta_T[:, 1, 3] = actions[:, 1]
                elif valid_action_mode == 'vx1':
                    actions[..., 0] = 0.5
                    actions[..., 1] = 0.
                    delta_T = torch.eye(4, device=actions.device, dtype=actions.dtype).unsqueeze(0).repeat(delta_T_origin.shape[0], 1, 1)
                    delta_T[:, 0, 3] = actions[:, 0]
                    delta_T[:, 1, 3] = actions[:, 1]
                elif valid_action_mode == 'vleft':
                    saved = torch.load("assets/traj/turn_left/66cdd6b659ae8be188468ebe.pt", weights_only=True)
                    actions = saved['actions']
                    delta_T = saved['delta_T']
                elif valid_action_mode == 'vright':
                    saved = torch.load("assets/traj/turn_right/66f14226424decfc585cd48c.pt", weights_only=True)
                    actions = saved['actions']
                    delta_T = saved['delta_T']
                elif valid_action_mode == 'hfstop':
                    stop_len = 8
                    t_hfstop = torch.linspace(0, 1, stop_len + 1, device=actions.device)[1:]
                    decay = 1 - (t_hfstop ** 2)
                    actions[16:16+stop_len] = actions[15] * decay.unsqueeze(1)
                    actions[16+stop_len:] = 0.
                    delta_T = torch.eye(4, device=actions.device, dtype=actions.dtype).unsqueeze(0).repeat(delta_T_origin.shape[0], 1, 1)
                    delta_T[:, 0, 3] = actions[:, 0]
                    delta_T[:, 1, 3] = actions[:, 1]
                elif valid_action_mode == 'moveleft':
                    delta_T = generate_lane_move(velocity=0.3, T_output=delta_T_origin.shape[0], lane_width=3.5, max_yaw_deg=20.0, direction='left').to(device=delta_T_origin.device, dtype=delta_T_origin.dtype)
                    actions[:, 0] = delta_T[:, 0, 3]
                    actions[:, 1] = delta_T[:, 1, 3]
                elif valid_action_mode == 'moveright':
                    delta_T = generate_lane_move(velocity=0.3, T_output=delta_T_origin.shape[0], lane_width=3.5, max_yaw_deg=20.0, direction='right').to(device=delta_T_origin.device, dtype=delta_T_origin.dtype)
                    actions[:, 0] = delta_T[:, 0, 3]
                    actions[:, 1] = delta_T[:, 1, 3]

                cur_T = torch.eye(4, device=actions.device, dtype=actions.dtype)
                vcs2world_save = [cur_T]
                for i in range(delta_T.shape[0]-1):
                    cur_T = cur_T @ delta_T[i]
                    vcs2world_save.append(cur_T.clone())

                vcs2world_save = torch.stack(vcs2world_save, dim=0)

        if "bbox" in conditions or "ttm" in conditions:
            masked_target = []
            ann_infos = [frame['ann_info'] for frame in frames]

            if valid_bbox_mode in ["default", "vx1", "moveleft", "moveright"]:
                bbox_ap_conds = nescenes_build_tracks_and_minmax_bboxes(
                    ann_infos,
                    pixel_values,
                    masked_target,
                    fill_value=np.nan,
                    clip_id=clip_id,
                    valid_mode=valid_mode,
                    conditions=conditions,
                    camera_names=self.camera_names,
                    delta_T=delta_T,
                    delta_T_origin=delta_T_origin,
                    camera_intrinsics=camera_intrinsics,
                    cam2lidar=camera2lidar,
                    cam2vcs=camera2ego,
                    lidar2vcs=lidar2ego,
                    valid_action_mode=valid_action_mode,
                    valid_bbox_mode=valid_bbox_mode,
                    valid_id_mode=valid_id_mode,
                )

                if "bbox" in conditions:
                    bbox_pixel_values = bbox_ap_conds["bbox_render"] * 2.0 - 1.0
                if "ttm" in conditions:
                    ref_ttm_value = bbox_ap_conds["ref_render"]
                    ref_ttm_vis = ref_ttm_value.detach().clone()
                if valid_id_mode != "default":
                    bbox_ap_conds_edit = nescenes_build_tracks_and_minmax_bboxes_apedit(
                        ann_infos,
                        pixel_values,
                        masked_target,
                        fill_value=np.nan,
                        clip_id=clip_id,
                        valid_mode=valid_mode,
                        conditions=conditions,
                        camera_names=self.camera_names,
                        delta_T=delta_T,
                        delta_T_origin=delta_T_origin,
                        camera_intrinsics=camera_intrinsics,
                        cam2lidar=camera2lidar,
                        cam2vcs=camera2ego,
                        lidar2vcs=lidar2ego,
                        valid_action_mode=valid_action_mode,
                        valid_bbox_mode=valid_bbox_mode,
                        valid_id_mode=valid_id_mode,
                    )

                    if "bbox" in conditions:
                        bbox_pixel_values = bbox_ap_conds_edit["bbox_render"] * 2.0 - 1.0
                    if "ttm" in conditions:
                        ref_ttm_value = bbox_ap_conds_edit["ref_render"]
                        ref_ttm_vis = 0.5 * ref_ttm_value.detach().clone() + 0.5 * (0.5 * pixel_values + 0.5)
                        ref_origin_ttm = bbox_ap_conds["ref_render"]

        text = 'A realistic dashcam video showing a driving scene. The environment and cars are distinct and highly detailed. Clear video quality.'
 
        sample = {
            "pixel_values": pixel_values,
            "text": text,
            "clip_id": clip_id,
            "data_type": 'video',
            "idx": idx,
            "ref_ttm_value": [],
            "ref_origin_ttm": [],
            "ref_ttm_vis": [],
            "conditions": {},
        }
        if bbox_pixel_values is not None:
            sample["conditions"]["bbox"] = bbox_pixel_values
        if ref_ttm_value is not None:
            sample["ref_ttm_value"] = ref_ttm_value
        if ref_origin_ttm is not None:
            sample["ref_origin_ttm"] = ref_origin_ttm
        if ref_ttm_vis is not None:
            sample["ref_ttm_vis"] = ref_ttm_vis
        if "action" in conditions:
            sample["conditions"]["action"] = actions
        return sample
    
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(
        self,
        kwargs,
    ) -> Dict[str, Any]:
        
        idx = kwargs["idx"]
        model_mode = kwargs["model_mode"]
        resolution = kwargs["resolution"]
        conditions = kwargs.get("conditions", [])
        num_condition_images = kwargs.get("num_condition_images", None)
        validation_mode = kwargs.get("validation_mode", None)

        if model_mode != "i2v" and num_condition_images is not None:
            raise ValueError(f"num_condition_images is not supported for {model_mode} model")

        assert model_mode in ["t2v", "i2v"], f"Invalid mode: {model_mode}"

        while True:
            sample = {}
            try:
                sample = self.get_batch(idx, resolution, conditions, valid_mode=validation_mode)
                if len(sample) > 0:
                    break
            
            except Exception as e:
                if validation_mode is None:
                    traceback.print_exc()
                    print(f"Error processing video: {e}")
                    idx = random.randint(0, self.length - 1)
                else:
                    print(f"Error processing video: {idx} {e}")
                    return None

        if model_mode == "i2v":
            mask_type = random.choices(list(self.i2v_random_mask_probs.keys()), weights=list(self.i2v_random_mask_probs.values()), k=1)[0]
            mask = self.get_mask(sample["pixel_values"].size(), mask_type, num_condition_images)
            mask_pixel_values = sample["pixel_values"] * (1 - mask)
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask
            
            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values
            
            ref_pixel_values = sample["pixel_values"][0].unsqueeze(0)
            if (mask == 1).all():
                ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            sample["ref_pixel_values"] = ref_pixel_values
        sample["model_mode"] = model_mode
        return sample
