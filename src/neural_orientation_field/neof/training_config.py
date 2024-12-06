from dataclasses import dataclass, field


@dataclass
class NeOFTrainingConfig:
    # Near/far clipping distance.
    nc: int = 1
    fc: int = 10
    # Positional encoding size.
    coarse_pos_encode: int = 6
    fine_pos_encode: int = 8
    # Camera ray sample rate.
    samples_per_ray: int = 16
    subd_samples: list[int] = field(
        default_factory=lambda: [8, 4, 2, 1])
    # Hyper parameters.
    lr: float = 2e-4
    num_epoch: int = 2
    ray_batch_size: int = 8192
    # Misc.
    size_train_ray = 1
    num_valid_image = 4
    valid_per_epoch: int = 16
