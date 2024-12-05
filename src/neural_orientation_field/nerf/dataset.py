import pathlib
from typing import Optional
import pycolmap

from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm

import neural_orientation_field.colmap.colmap_utils as colutils
from neural_orientation_field.nerf.utils import cam_ray_from_pose


class NeRFColmapImageDataset(Dataset):
    def __init__(self, image_path: pathlib.Path, model_path: pathlib.Path):
        self.image_path = image_path
        # Load COLMAP reconstruction.
        self.colmap_model = pycolmap.Reconstruction(model_path)
        # NeRF requires same camera.
        if self.colmap_model.num_cameras() != 1:
            raise ValueError(
                "Only COMAP reconstructions with single camera is accepted.")
        self.num_images: int = self.colmap_model.num_reg_images()
        self.cam_transforms, self.cam_params, self.image_file_names = colutils.get_camera_poses(
            self.colmap_model)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Convert image to (h, w, 3) np.ndarray.
        image = Image.open(self.image_path / self.image_file_names[idx])
        image = np.array(image) / 255
        h, w, _ = image.shape
        # Camera parameters.
        f, cx, cy = self.cam_params[idx]
        cam_transform = self.cam_transforms[idx]
        return image, cam_transform, (h, w), (f, cx, cy)


class NeRFImageDataset(Dataset):
    def __init__(self, frame_paths: list[pathlib.Path], param: np.ndarray, trans: np.ndarray):
        self.frame_paths = frame_paths
        f, cx, cy = param
        self.f = f
        self.cx = cx
        self.cy = cy
        self.cam_transforms = trans

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        # Convert image to (h, w, 3) np.ndarray.
        image = Image.open(self.frame_paths[idx])
        image = np.array(image) / 255
        h, w, _ = image.shape
        # Remove alpha channel.
        image = image[:, :, :3]
        cam_transform = np.linalg.inv(self.cam_transforms[idx])
        return image, cam_transform, (h, w), (self.f, self.cx, self.cy)


class NeRFRayDataset(Dataset):
    def __init__(self, image_dataset: Dataset, tqdm: Optional[tqdm] = None):
        self.pixels = []
        self.cam_origs = []
        self.cam_ray_worlds = []
        self.prefix_idx = []
        self.size = 0
        for image, cam_transform, (h, w), (f, cx, cy) in image_dataset:
            cam_orig, cam_ray_world = cam_ray_from_pose(
                cam_transform, h, w, f, cx, cy)
            if image.shape != cam_ray_world.shape:
                raise ValueError("image and cam_ray_world shape not match.")
            num_pixels = image.shape[0] * image.shape[1]
            image = image.reshape(-1, 3)
            cam_ray_world = cam_ray_world.reshape(-1, 3)
            self.pixels.append(image)
            self.cam_origs.append(cam_orig)
            self.cam_ray_worlds.append(cam_ray_world)
            self.prefix_idx.append(self.size)
            self.size += num_pixels
            if tqdm:
                tqdm.update(1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_idx = -1
        for prefix in self.prefix_idx:
            if prefix > idx:
                break
            image_idx += 1
        pixel_idx = idx - self.prefix_idx[image_idx]
        return self.cam_origs[image_idx], self.cam_ray_worlds[image_idx][pixel_idx], self.pixels[image_idx][pixel_idx]
