import torch

from neural_orientation_field.nerf.model import NerfModel


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


def volumetric_renderer(
    model: NerfModel,
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
    sample_pos = camera_origs[:, torch.newaxis, :] + camera_rays[:,
                                                                 torch.newaxis, :] * sample_depths[:, :, torch.newaxis]
    sample_direct = camera_rays.unsqueeze(1).expand((-1, num_sample, -1))

    # Send samples to NeRF.
    sample_pos = sample_pos.reshape(-1, 3)
    sample_direct = sample_direct.reshape(-1, 3)
    sample_pos_encode = pos_encode(sample_pos, num_pos_encode, device=device)

    # Eval radiance from NeRF
    nerf_input = torch.concat((sample_pos.unsqueeze(1), sample_direct.unsqueeze(
        1), sample_pos_encode), 1).reshape(n * num_sample, -1)
    radiance = model(nerf_input)
    radiance = radiance.reshape(n, num_sample, -1)

    # Integrate color.
    sample_depths_diff = sample_depths[:, 1:] - sample_depths[:, :-1]
    sample_depths_diff = torch.concat(
        (sample_depths_diff, torch.ones((n, 1), device=device) * interval), dim=1)
    color = radiance[:, :, 0:3] * radiance[:, :,
                                           3].unsqueeze(-1) * sample_depths_diff[:, :, torch.newaxis]
    color = torch.sum(color, 1)
    color = torch.sigmoid(color)
    return color
