import os
import gc
import imageio
import inspect
import importlib
import numpy as np
import torch
import time
import torchvision
import cv2
from einops import rearrange
from PIL import Image
from typing import Optional

from videox_fun.data.dataset_video import get_video_reader_batch
import torch.distributed as dist
from accelerate import PartialState
from accelerate.state import DistributedType


def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def get_width_and_height_from_image_and_base_resolution(image, base_resolution):
    target_pixels = int(base_resolution) * int(base_resolution)
    original_width, original_height = Image.open(image).size
    ratio = (target_pixels / (original_width * original_height)) ** 0.5
    width_slider = round(original_width * ratio)
    height_slider = round(original_height * ratio)
    return height_slider, width_slider

def color_transfer(sc, dc):
    """
    Transfer color distribution from of sc, referred to dc.

    Args:
        sc (numpy.ndarray): input image to be transfered.
        dc (numpy.ndarray): reference image

    Returns:
        numpy.ndarray: Transferred color distribution on the sc.
    """

    def get_mean_and_std(img):
        x_mean, x_std = cv2.meanStdDev(img)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    sc = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(sc)
    dc = cv2.cvtColor(dc, cv2.COLOR_RGB2LAB)
    t_mean, t_std = get_mean_and_std(dc)
    img_n = ((sc - s_mean) * (t_std / s_std)) + t_mean
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
    return dst

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=10, imageio_backend=True, color_transfer_post_process=False, control_videos: Optional[torch.Tensor] = None):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    
    # Process control_videos if provided
    if control_videos is not None:
        control_videos = rearrange(control_videos, "b c t h w -> t b c h w")
        
    
    outputs = []
    for i, x in enumerate(videos):
        # Concatenate control video frame if provided
        if control_videos is not None:
            control_x = control_videos[i]
            # Horizontally concatenate control video (left) with main video (right)
            x = torch.cat([control_x, x], dim=-1)  # concatenate along width dimension
        
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)   # shape [C, H, W]
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(Image.fromarray(x))

    if color_transfer_post_process:
        for i in range(1, len(outputs)):
            outputs[i] = Image.fromarray(color_transfer(np.uint8(outputs[i]), np.uint8(outputs[0])))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio_backend:
        if path.endswith("mp4"):
            imageio.mimsave(path, outputs,quality=8,codec='libx264', macro_block_size=10,fps=fps)
        else:
            imageio.mimsave(path, outputs, duration=(1000 * 1/fps))
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        outputs[0].save(path, format='GIF', append_images=outputs, save_all=True, duration=100, loop=0)




def get_multiview_tile_img(
    videos,
    camera_names=None,
    met3r_metric=None,
    fid_metric=None,
    fvd_metric=None,
    rescale=False,
):
    videos = rearrange(videos, "(n t) c h w -> t n c h w", n=len(camera_names))
    outputs = []
    for i, x in enumerate(videos):
        if rescale:
            x = (x + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)
        x = (x.float() * 255).detach().cpu().numpy().astype(np.uint8)   # (n,c,h,w)
        fvd_str = ""
        channel, landscape_height, landscape_width = x.shape[1], x.shape[2], x.shape[3]
        height = landscape_height * 3
        width = landscape_width * 3
        tiled_img = np.zeros((height, width, channel), dtype=np.uint8)
        for cam_id, cam_name in enumerate(camera_names):
            cam_img = x[cam_id].transpose(1, 2, 0)
            cam_img = np.ascontiguousarray(cam_img)
            if met3r_metric is not None:
                cv2.putText(cam_img, f"met3r: {met3r_metric[cam_id]:.6f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if fid_metric is not None:
                cv2.putText(cam_img, f"FID: {fid_metric[cam_id]:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if fvd_metric is not None:
                fvd_str += f"{fvd_metric[cam_id]:.2f} "

            if len(camera_names) == 1:
                tiled_img = cam_img
                height, width = landscape_height, landscape_width
                break

            if cam_name == "camera_front_30fov":
                # Place CAM_FRONT_30FOV at the top center
                tiled_img[:landscape_height, landscape_width:2*landscape_width, :] = cam_img
            elif cam_name == "camera_front_left":
                # Place CAM_FRONT_LEFT at the left center
                tiled_img[landscape_height:2*landscape_height, :landscape_width, :] = cam_img
            elif cam_name == "camera_front":
                # Place CAM_FRONT at the center
                tiled_img[landscape_height:2*landscape_height, landscape_width:2*landscape_width, :] = cam_img
            elif cam_name == "camera_front_right":
                # Place CAM_FRONT_RIGHT at the right center
                tiled_img[landscape_height:2*landscape_height, 2*landscape_width:, :] = cam_img
            elif cam_name == "camera_rear_left":
                # Place CAM_BACK_LEFT at the bottom left
                tiled_img[2*landscape_height:, :landscape_width, :] = cam_img
            elif cam_name == "camera_rear":
                # Place CAM_BACK at the bottom center
                tiled_img[2*landscape_height:, landscape_width:2*landscape_width, :] = cam_img
            elif cam_name == "camera_rear_right":
                # Place CAM_BACK_RIGHT at the bottom right
                tiled_img[2*landscape_height:, 2*landscape_width:, :] = cam_img
            
        if fvd_metric is not None:
            fvd_str = f"FVD: {fvd_str}"
            cv2.putText(tiled_img, fvd_str, (width // 2 - 100, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        outputs.append(tiled_img)
    return outputs


def save_multiview_videos_grid(
    videos: torch.Tensor,
    path: str,
    rescale=False,
    gt_videos: torch.Tensor= None,
    fps=10,
    imageio_backend=True,
    color_transfer_post_process=False,
    camera_names=None,
    met3r_metric=None,
    fid_metric=None,
    fvd_metric=None,
    overlap=False,
):
    """Combine cameras into a tiled image.
    Layout:
        ################################################################
        #                 #  CAM_FRONT_30FOV   #                       #
        ################################################################
        # CAM_FRONT_LEFT  #     CAM_FRONT      #     CAM_FRONT_RIGHT   #
        ################################################################
        #  CAM_BACK_LEFT  #     CAM_BACK       #     CAM_BACK_RIGHT    #
        ################################################################
    """
    videos_outputs = get_multiview_tile_img(
        videos,
        camera_names=camera_names,
        met3r_metric=met3r_metric,
        fid_metric=fid_metric,
        fvd_metric=fvd_metric,
        rescale=rescale,
    )
    if gt_videos is not None:
        gt_videos_outputs = get_multiview_tile_img(
            gt_videos,
            camera_names=camera_names,
            rescale=rescale,
        )
    outputs = []
    for i, tiled_img in enumerate(videos_outputs):
        if gt_videos is not None:
            gt_tiled_img = gt_videos_outputs[min(i, len(gt_videos_outputs) - 1)]
            if not overlap:
                tiled_img = np.concatenate([tiled_img, np.zeros((16, tiled_img.shape[1], tiled_img.shape[2]), dtype=np.uint8)], axis=0)
                tiled_img = np.concatenate([tiled_img, gt_tiled_img], axis=0)
            else:
                tiled_img = cv2.addWeighted(
                    tiled_img, 0.5, gt_tiled_img, 0.5, 0
                )
        outputs.append(Image.fromarray(tiled_img))

    if color_transfer_post_process:
        for i in range(1, len(outputs)):
            outputs[i] = Image.fromarray(color_transfer(np.uint8(outputs[i]), np.uint8(outputs[0])))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio_backend:
        if path.endswith("mp4"):
            imageio.mimsave(path, outputs,quality=8,codec='libx264', macro_block_size=10,fps=fps)
        else:
            imageio.mimsave(path, outputs, duration=(1000 * 1/fps))
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        outputs[0].save(path, format='GIF', append_images=outputs, save_all=True, duration=100, loop=0)

from PIL import ImageDraw
def save_met3r_debug_images(
    debug_infos,
    debug_met3r_dir_name,
):
    for debug_info in debug_infos:
        filename = os.path.join(debug_met3r_dir_name, f"{debug_info['flag']}.png")
        '''
        Create a 2x2 debug canvas with the score overlaid on top.
        Layout:
        image_1, image_2,
        overlap_mask, score_map
        '''
        image_1 = Image.fromarray(np.uint8(debug_info['image_1'].transpose(1, 2, 0) * 255))
        image_2 = Image.fromarray(np.uint8(debug_info['image_2'].transpose(1, 2, 0) * 255))
        overlap_mask = Image.fromarray(np.uint8(debug_info['overlap_mask'] * 255))
        # Convert the score map into a pseudo-RGB heatmap.
        score_map_bgr = cv2.applyColorMap((np.uint8(debug_info['score_map'] * 255)), cv2.COLORMAP_JET)
        score_map_rgb = cv2.cvtColor(score_map_bgr, cv2.COLOR_BGR2RGB)
        # Paint non-overlap regions white for readability.
        score_map_rgb[debug_info['overlap_mask'] < 0.5] = [255, 255, 255]

        score_map = Image.fromarray(score_map_rgb)
        score_map = score_map.resize((image_1.width, image_1.height), Image.BILINEAR)

        combined_image = Image.new('RGB', (image_1.width * 2, image_1.height * 2))
        combined_image.paste(image_1, (0, 0))
        combined_image.paste(image_2, (image_1.width, 0))
        combined_image.paste(overlap_mask.convert('L').convert('RGB'), (0, image_1.height))
        combined_image.paste(score_map, (image_1.width, image_1.height))
        # Draw the score on the composite image.
        draw = ImageDraw.Draw(combined_image)
        draw.text((10, 10), f"Score: {debug_info['score']:.4f}", fill=(255, 255, 255))
        combined_image.save(filename)


def get_first_n_frames_to_video_latent(video_path, first_n_frames, video_length, sample_size):
    """
    
    returns:
        input_video: (1, 3, video_length, sample_size[0], sample_size[1])  0~1
        input_video_mask: (1, 1, video_length, sample_size[0], sample_size[1])  0~255
        clip_image: PIL.Image.Image
    """
    # frames: List of np.ndarray, shape (h, w, 3)
    target_height, target_width = sample_size
    frames = get_video_reader_batch(video_path, np.arange(video_length), target_height=target_height, target_width=target_width)
    clip_image = frames[0]
    
    video = [torch.from_numpy(np.array(frame)).permute(2, 0, 1).unsqueeze(0) for frame in frames]  # list of [1, 3, h, w]
    video = torch.stack(video, dim=2).to(torch.float32) / 255  # [1, 3, first_n_frames, h, w]
    
    input_video = video.new_zeros([1, 3, video_length, target_height, target_width])  # [1, 3, video_length, target_height, target_width]
    input_video[:, :, :first_n_frames] = video[:, :, :first_n_frames]

    input_video_mask = torch.zeros_like(input_video[:, :1])
    input_video_mask[:, :, first_n_frames:] = 255

    clip_image = Image.fromarray(clip_image)

    return input_video, input_video_mask, clip_image, video


def debug_multiview_hdmap_videos(
    images: list,
    hdmaps: list,
    path: str,
    fps=10,
    imageio_backend=True,
    camera_names=None,
):
    """Combine cameras into a tiled image.
    Layout:
        ################################################################
        #                 #  CAM_FRONT_30FOV   #                       #
        ################################################################
        # CAM_FRONT_LEFT  #     CAM_FRONT      #     CAM_FRONT_RIGHT   #
        ################################################################
        #  CAM_BACK_LEFT  #     CAM_BACK       #     CAM_BACK_RIGHT    #
        ################################################################
    """
    num_cameras = len(camera_names)
    num_frames = len(images) // num_cameras
    
    landscape_height, landscape_width, channel = images[0].shape
    height = landscape_height * 3
    width = landscape_width * 3
    
    
    
    videos = videos[0]   # Assume batch size 1.
    videos = rearrange(videos, "c (n t) h w -> t n c h w", n=len(camera_names))
    
    outputs_img = []
    outputs_hdmap = []
    for frame_id in range(num_frames):
        tiled_img = np.zeros((height, width, channel), dtype=np.uint8)
        tiled_hdmap = np.zeros((height, width, channel), dtype=np.uint8)

        for cam_id, cam_name in enumerate(camera_names):
            cam_img = images[frame_id * num_cameras + cam_id].copy()
            cam_hdmap = hdmaps[frame_id * num_cameras + cam_id]
            cam_img[cam_hdmap > 0] = cam_hdmap[cam_hdmap > 0]

            if cam_name == "camera_front_30fov":
                # Place CAM_FRONT_30FOV at the top center
                tiled_img[:landscape_height, landscape_width:2*landscape_width, :] = cam_img
                tiled_hdmap[:landscape_height, landscape_width:2*landscape_width, :] = cam_hdmap
            elif cam_name == "camera_front_left":
                # Place CAM_FRONT_LEFT at the left center
                tiled_img[landscape_height:2*landscape_height, :landscape_width, :] = cam_img
                tiled_hdmap[landscape_height:2*landscape_height, :landscape_width, :] = cam_hdmap
            elif cam_name == "camera_front":
                # Place CAM_FRONT at the center
                tiled_img[landscape_height:2*landscape_height, landscape_width:2*landscape_width, :] = cam_img
                tiled_hdmap[landscape_height:2*landscape_height, landscape_width:2*landscape_width, :] = cam_hdmap
            elif cam_name == "camera_front_right":
                # Place CAM_FRONT_RIGHT at the right center
                tiled_img[landscape_height:2*landscape_height, 2*landscape_width:, :] = cam_img
                tiled_hdmap[landscape_height:2*landscape_height, 2*landscape_width:, :] = cam_hdmap
            elif cam_name == "camera_rear_left":
                # Place CAM_BACK_LEFT at the bottom left
                tiled_img[2*landscape_height:, :landscape_width, :] = cam_img
                tiled_hdmap[2*landscape_height:, :landscape_width, :] = cam_hdmap
            elif cam_name == "camera_rear":
                # Place CAM_BACK at the bottom center
                tiled_img[2*landscape_height:, landscape_width:2*landscape_width, :] = cam_img
                tiled_hdmap[2*landscape_height:, landscape_width:2*landscape_width, :] = cam_hdmap
            elif cam_name == "camera_rear_right":
                # Place CAM_BACK_RIGHT at the bottom right
                tiled_img[2*landscape_height:, 2*landscape_width:, :] = cam_img
        
        tiled_img = Image.fromarray(tiled_img)
        tiled_hdmap = Image.fromarray(tiled_hdmap)
        outputs_img.append(tiled_img)
        outputs_hdmap.append(tiled_hdmap)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio_backend:
        if path.endswith("mp4"):
            imageio.mimsave(path, outputs_img, quality=8,codec='libx264', macro_block_size=10,fps=fps)
            imageio.mimsave(path.replace('.mp4', '_hdmap.mp4'), outputs_hdmap, quality=8, codec='libx264', macro_block_size=10, fps=fps)
        else:
            imageio.mimsave(path, outputs_img, duration=(1000 * 1/fps))
            imageio.mimsave(path.replace('.mp4', '_hdmap.mp4'), outputs_hdmap, duration=(1000 * 1/fps))
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        outputs_img[0].save(path, format='GIF', append_images=outputs_img, save_all=True, duration=100, loop=0)
        outputs_hdmap[0].save(path.replace('.mp4', '_hdmap.mp4'), format='GIF', append_images=outputs_hdmap, save_all=True, duration=100, loop=0)


def get_image_to_video_latent(validation_image_start, validation_image_end, video_length, sample_size, mask_first_image=False):
    if validation_image_start is not None and validation_image_end is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]

        if type(validation_image_end) is str and os.path.isfile(validation_image_end):
            image_end = Image.open(validation_image_end).convert("RGB")
            image_end = image_end.resize([sample_size[1], sample_size[0]])
        else:
            image_end = validation_image_end
            image_end = [_image_end.resize([sample_size[1], sample_size[0]]) for _image_end in image_end]

        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255

        if type(image_end) is list:
            image_end = [_image_end.resize(image_start[0].size if type(image_start) is list else image_start.size) for _image_end in image_end]
            end_video = torch.cat(
                [torch.from_numpy(np.array(_image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_end in image_end], 
                dim=2
            )
            input_video[:, :, -len(end_video):] = end_video
            
            input_video_mask[:, :, -len(image_end):] = 0
        else:
            image_end = image_end.resize(image_start[0].size if type(image_start) is list else image_start.size)
            input_video[:, :, -1:] = torch.from_numpy(np.array(image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
            input_video_mask[:, :, -1:] = 0

        input_video = input_video / 255

    elif validation_image_start is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]
        image_end = None
        
        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            input_video = input_video / 255
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            ) / 255
            input_video_mask = torch.zeros_like(input_video[:, :1])
            if mask_first_image:
                input_video_mask[:, :, :] = 255
                clip_image = np.zeros_like(np.array(clip_image))
                clip_image = Image.fromarray(clip_image)
            else:
                input_video_mask[:, :, 1:] = 255
    else:
        image_start = None
        image_end = None
        input_video = torch.zeros([1, 3, video_length, sample_size[0], sample_size[1]])
        input_video_mask = torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]) * 255
        clip_image = None

    del image_start
    del image_end
    gc.collect()

    return  input_video, input_video_mask, clip_image

def get_video_to_video_latent(input_video_path, video_length, sample_size, fps=None, validation_video_mask=None, ref_image=None):
    if input_video_path is not None:
        if isinstance(input_video_path, str):
            cap = cv2.VideoCapture(input_video_path)
            input_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else max(1,int(original_fps // fps))

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            input_video = input_video_path

        input_video = torch.from_numpy(np.array(input_video))[:video_length]
        input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

        if validation_video_mask is not None:
            validation_video_mask = Image.open(validation_video_mask).convert('L').resize((sample_size[1], sample_size[0]))
            input_video_mask = np.where(np.array(validation_video_mask) < 240, 0, 255)
            
            input_video_mask = torch.from_numpy(np.array(input_video_mask)).unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
            input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
            input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)
        else:
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, :] = 255
    else:
        input_video, input_video_mask = None, None

    if ref_image is not None:
        if isinstance(ref_image, str):
            clip_image = Image.open(ref_image).convert("RGB")
        else:
            clip_image = Image.fromarray(np.array(ref_image, np.uint8))
    else:
        clip_image = None

    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        else:
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
    return input_video, input_video_mask, ref_image, clip_image

def get_image_latent(ref_image=None, sample_size=None):
    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        else:
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255

    return ref_image

def timer(func):
    def wrapper(*args, **kwargs):
        start_time  = time.time()
        result      = func(*args, **kwargs)
        end_time    = time.time()
        print(f"function {func.__name__} running for {end_time - start_time} seconds")
        return result
    return wrapper

def timer_record(model_name=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start_time = time.time()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            end_time = time.time()
            import torch.distributed as dist
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    time_sum  = end_time - start_time
                    print('# --------------------------------------------------------- #')
                    print(f'#   {model_name} time: {time_sum}s')
                    print('# --------------------------------------------------------- #')
                    # _write_to_excel(model_name, time_sum)
            else:
                time_sum  = end_time - start_time
                print('# --------------------------------------------------------- #')
                print(f'#   {model_name} time: {time_sum}s')
                print('# --------------------------------------------------------- #')
                # _write_to_excel(model_name, time_sum)
            return result
        return wrapper
    return decorator

def _write_to_excel(model_name, time_sum):
    import pandas as pd
    import os

    row_env = os.environ.get(f"{model_name}_EXCEL_ROW", "1")  # Default row 1.
    col_env = os.environ.get(f"{model_name}_EXCEL_COL", "1")  # Default column A.
    file_path = os.environ.get(f"EXCEL_FILE", "timing_records.xlsx")  # Default file name.

    try:
        df = pd.read_excel(file_path, sheet_name="Sheet1", header=None)
    except FileNotFoundError:
        df = pd.DataFrame()

    row_idx = int(row_env)
    col_idx = int(col_env)

    if row_idx >= len(df):
        df = pd.concat([df, pd.DataFrame([ [None] * (len(df.columns) if not df.empty else 0) ] * (row_idx - len(df) + 1))], ignore_index=True)

    if col_idx >= len(df.columns):
        df = pd.concat([df, pd.DataFrame(columns=range(len(df.columns), col_idx + 1))], axis=1)

    df.iloc[row_idx, col_idx] = time_sum

    df.to_excel(file_path, index=False, header=False, sheet_name="Sheet1")


def import_cls(type):
    """
    Import class based on its full path.
    
    Args:
        type (str): Full path of the class, e.g., 'module.submodule.ClassName'.
    
    Returns:
        class: The imported class.
    """
    module, cls = type.rsplit('.', 1)
    module = importlib.import_module(module, package=None)
    cls = getattr(module, cls)
    return cls


def construct_emb_cls(emb_info, additional_kwargs={}):
    """
    Construct embedding class based on provided information.
    
    Args:
        emb_info (dict): Information about the embedding, including 'name' and 'type'.
        additional_kwargs(dict): Additional arguments for the embedding class.
    
    Returns:
        nn.Module: An instance of the specified embedding class.
    """
    emb_cls = import_cls(emb_info['type'])
    all_kwargs = emb_info.get('kwargs', {})
    all_kwargs.update(additional_kwargs)
    return emb_cls(**all_kwargs)


def custom_gather_object(obj):
    """
    Compatibility wrapper for gathering Python objects across devices.
    """
    state = PartialState()
    
    if state.distributed_type == DistributedType.NO:
        # Return the object directly in non-distributed mode.
        return [obj]
    elif state.distributed_type == DistributedType.XLA:
        raise NotImplementedError("gather objects in TPU is not supported")
    else:
        # Use torch.distributed object gathering everywhere else.
        output = [None] * state.num_processes
        dist.all_gather_object(output, obj)
        return output
