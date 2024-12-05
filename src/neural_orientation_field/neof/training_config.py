from dataclasses import dataclass


@dataclass
class NeOFTrainingConfig:
    # Near/far clipping distance.
    nc: int = 1
    fc: int = 8
    # Positional encoding size.
    coarse_pos_encode: int = 4
    fine_pos_encode: int = 6
    # Camera ray sample rate.
    samples_per_ray: int = 6
    max_subd_samples: int = 8
    # Hyper parameters.
    lr: float = 2e-4
    num_epoch: int = 2
    ray_batch_size: int = 8192
    # Misc.
    size_train_ray = 1
    num_valid_image = 2
    valid_per_epoch: int = 16
