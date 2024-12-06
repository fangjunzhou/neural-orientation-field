import numpy as np
import torch

from neural_orientation_field.neof.model import NeOFCoarseModel, NeOFFineModel, NeOFModel
from neural_orientation_field.nerf.utils import pos_encode


def hair_dir_color2vec(hair_dir_col: np.ndarray):
    """Convert hair direction image to vector array.

    Args:
        hair_dir_col: colored hair direction image.

    Returns:
        screen space hair direction with positive x to the right and postive y to the top.
    """
    hair_vec = hair_dir_col[:, :, :2] * 2 - 1
    return hair_vec


def hair_dir_vec2color(hair_dir_vec: np.ndarray):
    """Convert hair direction image to vector array.

    Args:
        hair_dir_col: colored hair direction image.

    Returns:
        screen space hair direction with positive x to the right and postive y to the top.
    """
    h, w, _ = hair_dir_vec.shape
    hair_dir_color = (hair_dir_vec + 1) / 2
    hair_dir_color = np.concatenate(
        (hair_dir_color, np.zeros((h, w, 1))), axis=2)
    return hair_dir_color


def nerf_image_render(
        coarse_model: NeOFCoarseModel,
        fine_model: NeOFFineModel,
        cam_trans_np: np.ndarray,
        cam_orig_np: np.ndarray,
        cam_ray_np: np.ndarray,
        ray_batch_size: int,
        nc: float,
        fc: float,
        samples_per_ray: int,
        subd_samples: list[int],
        coarse_pos_encode: int,
        fine_pos_encode: int,
        device: torch.device = torch.device("cpu")
):
    h, w, _ = cam_ray_np.shape
    cam_trans = torch.from_numpy(cam_trans_np).type(torch.float32).to(device)
    cam_trans = cam_trans.view(1, 4, 4)
    cam_trans = cam_trans.expand((h * w, 4, 4))
    cam_orig = torch.from_numpy(cam_orig_np).type(torch.float32).to(device)
    cam_orig = cam_orig.view(1, 3)
    cam_orig = cam_orig.expand((h * w, 3))
    cam_ray = torch.from_numpy(cam_ray_np).type(torch.float32).to(device)
    cam_ray = cam_ray.reshape(-1, 3)
    num_pixels = cam_orig.shape[0]
    coarse_pred = torch.zeros((num_pixels, 2))
    fine_pred = torch.zeros((num_pixels, 2))
    for i in range(0, num_pixels, ray_batch_size):
        coarse_color_batch, occupancy, sample_depth = static_volumetric_renderer(
            coarse_model,
            cam_trans[i:i+ray_batch_size],
            cam_orig[i:i+ray_batch_size],
            cam_ray[i:i+ray_batch_size],
            nc,
            fc,
            num_sample=samples_per_ray,
            num_pos_encode=coarse_pos_encode,
            device=device
        )
        occupancy_body = occupancy[:, :, 0] + occupancy[:, :, 1]
        fine_color_batch, _, _ = adaptive_volumetric_renderer(
            fine_model,
            cam_trans[i:i+ray_batch_size],
            cam_orig[i:i+ray_batch_size],
            cam_ray[i:i+ray_batch_size],
            occupancy_body,
            sample_depth,
            subd_samples=subd_samples,
            num_pos_encode=fine_pos_encode,
            device=device
        )
        coarse_pred[i:i+ray_batch_size] = coarse_color_batch.detach()
        fine_pred[i:i+ray_batch_size] = fine_color_batch.detach()
    coarse_pred = coarse_pred.reshape((h, w, -1))
    fine_pred = fine_pred.reshape((h, w, -1))
    return coarse_pred, fine_pred


def static_volumetric_renderer(
    model: NeOFModel,
    cam_trans: torch.Tensor,
    camera_origs: torch.Tensor,
    camera_rays: torch.Tensor,
    nc: float,
    fc: float,
    num_sample: int,
    sample_jitter: float = 0.1,
    num_pos_encode: int = 6,
    device: torch.device = torch.device("cpu")
):
    """NeRF volumetric renderer.

    Args:
        model: NeRF model to use.
        camera_origs: camera origins.
        camera_rays: camera ray directions.
        nc: near clipping distance.
        fc: far clipping distance.
        num_sample: sample num for each camera ray.
        sample_jitter: sample jitter amount.
        num_pos_encode: number of position encoding to use.
        device: device to use.

    Returns:
        rendered pixels with the same size as camera_rays.
    """
    n, _ = camera_origs.shape
    # Expand ray to multiple samples.
    sample_depths = torch.linspace(nc, fc, num_sample, device=device)
    sample_depths = sample_depths.unsqueeze(0).expand(n, -1)
    # Improve convergence by introducing random sample.
    interval = (fc - nc) / (num_sample - 1)
    sample_depths = sample_depths + \
        (torch.rand(sample_depths.shape, device=device) - 0.5) * \
        interval * sample_jitter

    # Sample from NeRF.
    orientation = sample_orientation(
        model, camera_origs, camera_rays, sample_depths[:, :-1], num_pos_encode, device)

    # Integrate orientation.
    sample_depths_diff = sample_depths[:, 1:] - sample_depths[:, :-1]
    occupancy = torch.nn.functional.relu(orientation[:, :, 3:5])
    occupancy_hair = occupancy[:, :, 0]
    occupancy_face = occupancy[:, :, 1]
    occupancy_body = (occupancy_hair + occupancy_face) * sample_depths_diff
    residual_ray = torch.exp(-torch.cumsum(occupancy_body, dim=-1))
    # Transform to screen space.
    cam_trans = cam_trans.transpose(-1, -2)
    screen_space_orientation = torch.matmul(
        orientation[:, :, 0:3],
        cam_trans[:, :3, :3]
    )
    screen_space_orientation = torch.sigmoid(screen_space_orientation) * 2 - 1
    # Discard depth.
    screen_space_orientation = screen_space_orientation[:, :, :2]
    # Integrate.
    curr_occupancy = 1 - torch.exp(-occupancy_hair * sample_depths_diff)
    screen_space_orientation = residual_ray.unsqueeze(
        -1) * curr_occupancy.unsqueeze(-1) * screen_space_orientation
    screen_space_orientation = torch.sum(screen_space_orientation, 1)
    inv_hair_mask = torch.exp(-torch.sum(occupancy_hair *
                              sample_depths_diff, dim=-1))
    screen_space_orientation += inv_hair_mask.unsqueeze(-1) * torch.tensor(
        [-1, -1]).to(device).unsqueeze(0)
    return screen_space_orientation, occupancy, sample_depths


def adaptive_volumetric_renderer(
    model: NeOFModel,
    cam_trans: torch.Tensor,
    camera_origs: torch.Tensor,
    camera_rays: torch.Tensor,
    occupancy: torch.Tensor,
    sample_depths: torch.Tensor,
    subd_samples: list[int] = [4, 2],
    num_pos_encode: int = 6,
    device: torch.device = torch.device("cpu")
):
    sample_depths = adaptive_sample_depth(
        occupancy,
        sample_depths,
        subd_samples,
        device
    )
    # Sample from NeRF.
    orientation = sample_orientation(
        model, camera_origs, camera_rays, sample_depths[:, :-1], num_pos_encode, device)

    # Integrate orientation.
    sample_depths_diff = sample_depths[:, 1:] - sample_depths[:, :-1]
    occupancy = torch.nn.functional.relu(orientation[:, :, 3:5])
    occupancy_hair = occupancy[:, :, 0]
    occupancy_face = occupancy[:, :, 1]
    occupancy_body = (occupancy_hair + occupancy_face) * sample_depths_diff
    residual_ray = torch.exp(-torch.cumsum(occupancy_body, dim=-1))
    # Transform to screen space.
    cam_trans = cam_trans.transpose(-1, -2)
    screen_space_orientation = torch.matmul(
        orientation[:, :, 0:3],
        cam_trans[:, :3, :3]
    )
    # Discard depth.
    screen_space_orientation = screen_space_orientation[:, :, :2]
    screen_space_orientation = torch.sigmoid(screen_space_orientation) * 2 - 1
    # Integrate.
    curr_occupancy = 1 - torch.exp(-occupancy_hair * sample_depths_diff)
    screen_space_orientation = residual_ray.unsqueeze(
        -1) * curr_occupancy.unsqueeze(-1) * screen_space_orientation
    screen_space_orientation = torch.sum(screen_space_orientation, 1)

    inv_hair_mask = torch.exp(-torch.sum(occupancy_hair *
                              sample_depths_diff, dim=-1))
    screen_space_orientation += inv_hair_mask.unsqueeze(-1) * torch.tensor(
        [-1, -1]).to(device).unsqueeze(0)

    return screen_space_orientation, occupancy, sample_depths


def adaptive_sample_depth(
    occupancy: torch.Tensor,
    sample_depths: torch.Tensor,
    subd_samples: list[int] = [4, 2],
    device: torch.device = torch.device("cpu")
):
    """Generate adaptive sample depth from depth and occupancy.

    Args:
        occupancy: occupancy to the corresponding depth.
        sample_depths: sample depths.
        num_subdiv_sample: maximum number of subdivision to use.

    Returns:
        subdivided sample depth.
    """
    batch_size, _ = sample_depths.shape
    # Get the occupancy rank.
    _, occupancy_rank = torch.sort(occupancy[:, :-1], dim=-1, descending=True)
    for i in range(min(len(subd_samples), occupancy_rank.shape[1])):
        depth_idx = occupancy_rank[:, i]
        subd = subd_samples[i]
        depth_start = sample_depths[torch.arange(
            0, batch_size), depth_idx].unsqueeze(1)
        depth_end = sample_depths[torch.arange(
            0, batch_size), depth_idx + 1].unsqueeze(1)
        steps = torch.linspace(0, 1, subd + 2).unsqueeze(0).to(device)
        subd_depth = depth_start + (depth_end - depth_start) * steps
        sample_depths = torch.concat(
            [sample_depths, subd_depth[:, 1:-1]], dim=-1)
    sample_depths, _ = torch.sort(sample_depths, dim=-1)
    return sample_depths


def sample_orientation(
    model: NeOFModel,
    camera_origs: torch.Tensor,
    camera_rays: torch.Tensor,
    sample_depths: torch.Tensor,
    num_pos_encode: int = 6,
    device: torch.device = torch.device("cpu")
):
    """Sample radiance from NeRF model.

    Args:
        model: NeRF model.
        camera_origs: camera origins.
        camera_rays: camera ray directions.
        num_sample: sample num for each camera ray.
        num_pos_encode: number of position encoding to use.
        device: device to use.

    Returns:
        radiance tensor.
    """
    batch_size, _ = camera_origs.shape
    num_sample = sample_depths.shape[1]
    # Get sample positions and directions.
    sample_pos = camera_origs[:, torch.newaxis, :] + camera_rays[:,
                                                                 torch.newaxis, :] * sample_depths[:, :, torch.newaxis]
    sample_direct = camera_rays.unsqueeze(1).expand((-1, num_sample, -1))
    # Send samples to NeRF.
    sample_pos = sample_pos.reshape(-1, 3)
    sample_direct = sample_direct.reshape(-1, 3)
    sample_pos_encode = pos_encode(sample_pos, num_pos_encode, device=device)
    # Eval radiance from NeRF
    nerf_input = torch.concat((sample_pos.unsqueeze(1), sample_direct.unsqueeze(
        1), sample_pos_encode), 1).reshape(batch_size * num_sample, -1)
    orientation = model(nerf_input)
    orientation = orientation.reshape(batch_size, num_sample, -1)
    return orientation
