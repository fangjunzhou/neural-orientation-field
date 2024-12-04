from dataclasses import dataclass


@dataclass
class NeRFTrainingConfig:
    # Near/far clipping distance.
    nc: int = 1
    fc: int = 8
    # Positional encoding size.
    coarse_pos_encode: int = 2
    fine_pos_encode: int = 4
    # Camera ray sample rate.
    samples_per_ray: int = 4
    max_subd_samples: int = 8
    # Hyper parameters.
    lr: float = 2e-4
    num_epoch: int = 8
    ray_batch_size: int = 8192
    # Misc.
    num_valid_image = 4
    valid_per_epoch: int = 16
