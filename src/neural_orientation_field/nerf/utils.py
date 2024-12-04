import torch
import numpy as np

from neural_orientation_field.nerf.model import NeRFModel, NeRfCoarseModel, NeRfFineModel


def pos_encode(x: torch.Tensor, num_encoding_functions: int = 6, device: torch.device = torch.device("cpu")):
    """Position encoding function for NeRF.

    Args:
        x: (n, 3) position to encode.
        num_encoding_functions: number of exponent to use.
        device: device to use.

    Returns:
        (n, num_encoding_functions, 3) encoded positions
    """
    p = torch.arange(num_encoding_functions, device=device)
    x = torch.pow(2, p)[torch.newaxis, :, torch.newaxis] * \
        x[:, torch.newaxis, :]
    x = torch.concat((torch.sin(x), torch.cos(x)), 1)
    return x


def nerf_image_render(
        coarse_model: NeRfCoarseModel,
        fine_model: NeRfFineModel,
        cam_orig_np: np.ndarray,
        cam_ray_np: np.ndarray,
        ray_batch_size: int,
        nc: float,
        fc: float,
        samples_per_ray: int,
        max_subd_samples: int,
        coarse_pos_encode: int,
        fine_pos_encode: int,
        device: torch.device = torch.device("cpu")
):
    h, w, _ = cam_ray_np.shape
    cam_orig = torch.from_numpy(cam_orig_np).type(torch.float32).to(device)
    cam_orig = cam_orig.view(1, 1, -1)
    cam_orig = cam_orig.expand(cam_ray_np.shape)
    cam_ray = torch.from_numpy(cam_ray_np).type(torch.float32).to(device)
    cam_orig = cam_orig.reshape(-1, 3)
    cam_ray = cam_ray.reshape(-1, 3)
    num_pixels = cam_orig.shape[0]
    coarse_pred = torch.zeros((num_pixels, 3))
    fine_pred = torch.zeros((num_pixels, 3))
    for i in range(0, num_pixels, ray_batch_size):
        coarse_color_batch, occupancy, sample_depth = static_volumetric_renderer(
            coarse_model,
            cam_orig[i:i+ray_batch_size],
            cam_ray[i:i+ray_batch_size],
            nc,
            fc,
            num_sample=samples_per_ray,
            num_pos_encode=coarse_pos_encode,
            device=device
        )
        fine_color_batch, _, _ = adaptive_volumetric_renderer(
            fine_model,
            cam_orig[i:i+ray_batch_size],
            cam_ray[i:i+ray_batch_size],
            occupancy,
            sample_depth,
            max_subd_sample=max_subd_samples,
            num_pos_encode=fine_pos_encode,
            device=device
        )
        coarse_pred[i:i+ray_batch_size] = coarse_color_batch.detach()
        fine_pred[i:i+ray_batch_size] = fine_color_batch.detach()
    coarse_pred = coarse_pred.reshape((h, w, -1))
    fine_pred = fine_pred.reshape((h, w, -1))
    return coarse_pred, fine_pred


def static_volumetric_renderer(
    model: NeRFModel,
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
    radiance = sample_radiance(
        model, camera_origs, camera_rays, sample_depths[:, :-1], num_pos_encode, device)

    # Integrate color.
    sample_depths_diff = sample_depths[:, 1:] - sample_depths[:, :-1]
    occupancy = radiance[:, :, 3]
    color = radiance[:, :, 0:3] * \
        occupancy.unsqueeze(-1) * sample_depths_diff[:, :, torch.newaxis]
    color = torch.sum(color, 1)
    color = torch.sigmoid(color)
    return color, occupancy, sample_depths


def adaptive_volumetric_renderer(
    model: NeRFModel,
    camera_origs: torch.Tensor,
    camera_rays: torch.Tensor,
    occupancy: torch.Tensor,
    sample_depths: torch.Tensor,
    max_subd_sample: int = 2,
    num_pos_encode: int = 6,
    device: torch.device = torch.device("cpu")
):
    sample_depths = adaptive_sample_depth(
        occupancy,
        sample_depths,
        max_subd_sample,
        device
    )
    # Sample from NeRF.
    radiance = sample_radiance(
        model, camera_origs, camera_rays, sample_depths[:, :-1], num_pos_encode, device)

    # Integrate color.
    sample_depths_diff = sample_depths[:, 1:] - sample_depths[:, :-1]
    occupancy = torch.sigmoid(radiance[:, :, 3])
    color = radiance[:, :, 0:3] * \
        occupancy.unsqueeze(-1) * sample_depths_diff[:, :, torch.newaxis]
    color = torch.sum(color, 1)
    color = torch.sigmoid(color)
    return color, occupancy, sample_depths


def adaptive_sample_depth(
    occupancy: torch.Tensor,
    sample_depths: torch.Tensor,
    max_subd_sample: int = 2,
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
    for i in range(min(max_subd_sample, occupancy_rank.shape[1])):
        depth_idx = occupancy_rank[:, i]
        subd = max_subd_sample - i
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


def sample_radiance(
    model: NeRFModel,
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
    radiance = model(nerf_input)
    radiance = radiance.reshape(batch_size, num_sample, -1)
    return radiance


def cam_ray_from_pose(cam_transform: np.ndarray, h: int, w: int, f: float, cx: float, cy: float):
    """Get camera orgin and camera ray from its pose.

    Args:
        cam_transform: World to view coordinate transform.
        h: image height.
        w: image width.
        f: focal length.
        cx: focal point x.
        cy: focal point y.

    Returns:
        camera origins and camera rays.
    """
    # Camera pose.
    cam_transform_inv = np.linalg.inv(cam_transform)
    # Calculate camera origins
    cam_orig = np.matmul(cam_transform_inv, np.array([0, 0, 0, 1]))[:3]
    # Calculate camera ray.
    pixel_coord = np.moveaxis(
        np.mgrid[0:h, 0:w], [0, 1, 2], [2, 1, 0]) - np.array([cx, cy])
    cam_ray_view = np.append(pixel_coord, -f * np.ones((h, w, 1)), axis=2)
    cam_ray_view_homo = np.append(
        cam_ray_view, np.zeros((h, w, 1)), axis=2)
    cam_ray_world: np.ndarray = np.matmul(
        cam_transform_inv[np.newaxis, np.newaxis, :, :],
        cam_ray_view_homo[:, :, :, np.newaxis]
    ).reshape((h, w, -1))[:, :, :3]
    cam_ray_world = cam_ray_world / \
        np.linalg.norm(cam_ray_world, axis=-1)[:, :, np.newaxis]

    return cam_orig, cam_ray_world
