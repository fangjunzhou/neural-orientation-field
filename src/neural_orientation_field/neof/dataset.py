import pathlib
from typing import Optional

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from neural_orientation_field.neof.utils import hair_dir_color2vec
from neural_orientation_field.nerf.utils import cam_ray_from_pose


class NeOFImageDataset(Dataset):
    def __init__(
            self,
            body_mask_paths: list[pathlib.Path],
            hair_mask_path: list[pathlib.Path],
            hair_dir_paths: list[pathlib.Path],
            param: np.ndarray,
            trans: np.ndarray
    ):
        self.body_mask_paths = body_mask_paths
        self.hair_mask_path = hair_mask_path
        self.hair_dir_paths = hair_dir_paths
        assert len(body_mask_paths) == len(hair_mask_path) and len(
            hair_mask_path) == len(hair_dir_paths)
        f, cx, cy = param
        self.f = f
        self.cx = cx
        self.cy = cy
        self.cam_transforms = trans

    def __len__(self):
        return len(self.hair_dir_paths)

    def __getitem__(self, idx):
        # Convert image to (h, w, 3) np.ndarray.
        body_mask = Image.open(self.body_mask_paths[idx])
        body_mask = np.array(body_mask)[:, :, 0] / 255
        hair_mask = Image.open(self.hair_mask_path[idx])
        hair_mask = np.array(hair_mask)[:, :, 0] / 255
        hair_dir = Image.open(self.hair_dir_paths[idx])
        hair_dir = np.array(hair_dir) / 255
        hair_dir = hair_dir_color2vec(hair_dir[:, :, :3])

        h, w, _ = hair_dir.shape

        cam_transform = np.linalg.inv(self.cam_transforms[idx])

        return (body_mask, hair_mask, hair_dir), cam_transform, (h, w), (self.f, self.cx, self.cy)


class NeOFRayDataset(Dataset):
    def __init__(self, image_dataset: Dataset, tqdm: Optional[tqdm] = None):
        self.body_masks = []
        self.hair_masks = []
        self.hair_dirs = []
        self.cam_transforms = []
        self.cam_origs = []
        self.cam_ray_worlds = []
        self.prefix_idx = []
        self.size = 0
        for (body_mask, hair_mask, hair_dir), cam_transform, (h, w), (f, cx, cy) in image_dataset:
            cam_orig, cam_ray_world = cam_ray_from_pose(
                cam_transform, h, w, f, cx, cy)
            num_pixels = hair_dir.shape[0] * hair_dir.shape[1]
            body_mask = body_mask.reshape(-1, 1)
            hair_mask = hair_mask.reshape(-1, 1)
            hair_dir = hair_dir.reshape(-1, 2)
            cam_ray_world = cam_ray_world.reshape(-1, 3)
            self.cam_transforms.append(cam_transform)
            self.body_masks.append(body_mask)
            self.hair_masks.append(hair_mask)
            self.hair_dirs.append(hair_dir)
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
        return self.cam_transforms[image_idx], self.cam_origs[image_idx], self.cam_ray_worlds[image_idx][pixel_idx], self.body_masks[image_idx][pixel_idx], self.hair_masks[image_idx][pixel_idx], self.hair_dirs[image_idx][pixel_idx]
