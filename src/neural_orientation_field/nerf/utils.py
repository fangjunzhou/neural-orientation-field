import torch

from neural_orientation_field.nerf.model import NeRFModel


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
    for i in range(max_subd_sample):
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
