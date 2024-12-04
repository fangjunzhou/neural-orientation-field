import argparse
import pathlib
import logging
import sys
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from neural_orientation_field.nerf.dataset import NeRFPriorImageDataset, NeRFRayDataset
from neural_orientation_field.nerf.model import NeRfCoarseModel, NeRfFineModel
from neural_orientation_field.nerf.utils import cam_ray_from_pose, nerf_image_render,  static_volumetric_renderer, adaptive_volumetric_renderer
from neural_orientation_field.nerf.training_config import NeRFTrainingConfig


def main():
    logging.basicConfig(level=logging.INFO)
    # ---------------------- Argument Setup ---------------------- #
    parser = argparse.ArgumentParser(
        prog="NeRF Trainer",
        description="""
        NeRF trainer.
        """
    )
    parser.add_argument(
        "-i",
        "--image",
        default=pathlib.Path("./data/images/"),
        help="""
        Input images path.
        """,
        type=pathlib.Path
    )
    parser.add_argument(
        "-c",
        "--cam",
        default=pathlib.Path("./data/camera/"),
        help="""
        Input camera pose path.
        """,
        type=pathlib.Path
    )
    parser.add_argument(
        "-o",
        "--output",
        default=pathlib.Path("./data/output/nerf/model/"),
        help="""
        Output model directory.
        """,
        type=pathlib.Path
    )
    parser.add_argument(
        "-e",
        "--eval",
        default=pathlib.Path("./data/output/nerf/eval/"),
        help="""
        Output evaluation directory.
        """,
        type=pathlib.Path
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cpu",
        help="""
        Hardware acceleration device.
        """,
        type=str
    )
    # ------------------- Read Project Config  ------------------- #
    args = parser.parse_args()
    image_path: pathlib.Path = args.image
    if not image_path.exists():
        logging.error("The input image directory doesn't exist.")
        sys.exit(1)
    cam_path: pathlib.Path = args.cam
    if not cam_path.exists():
        logging.error("The input camera pose directory doesn't exist.")
        sys.exit(1)
    output_path: pathlib.Path = args.output
    if not output_path.exists():
        output_path.mkdir(parents=True)
    eval_path: pathlib.Path = args.eval
    if not eval_path.exists():
        eval_path.mkdir(parents=True)

    device_arg = args.device
    if device_arg == "mps" and torch.mps.is_available():
        device = torch.device("mps")
    elif device_arg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device {device}")

    config = NeRFTrainingConfig()

    # ----------------------- Load Dataset ----------------------- #

    # Load camera parameters.
    frame_name_path = cam_path / "frame-names.txt"
    cam_transform_path = cam_path / "camera-transforms.npy"
    cam_param_path = cam_path / "camera-params.npy"
    with open(frame_name_path, "r") as frame_path_file:
        frame_names = frame_path_file.read().split("\n")
        frame_paths = [image_path / frame_name for frame_name in frame_names]
    with open(cam_transform_path, "rb") as cam_transform_file:
        cam_transforms = np.load(cam_transform_file)
    with open(cam_param_path, "rb") as cam_param_file:
        cam_params = np.load(cam_param_file)

    # Load image dataset.
    image_dataset = NeRFPriorImageDataset(
        frame_paths, cam_params, cam_transforms)
    num_train = int(config.train_test_split * len(image_dataset))
    num_test = len(image_dataset) - num_train
    image_dataset_train, image_dataset_test = random_split(
        image_dataset, [num_train, num_test])
    logging.info(f"Training image size: {len(image_dataset_train)}")
    logging.info(f"Testing image size: {len(image_dataset_test)}")
    # Loading ray dataset.
    with tqdm(total=len(image_dataset_train), desc="Processing Image") as progress:
        ray_dataset_train = NeRFRayDataset(image_dataset_train, progress)

    # ------------------------- Training ------------------------- #

    # Init model.
    coarse_model = NeRfCoarseModel(
        num_encoding_functions=config.coarse_pos_encode)
    coarse_model.to(device)
    coarse_optimizer = torch.optim.Adam(
        coarse_model.parameters(), lr=config.lr)

    fine_model = NeRfFineModel(num_encoding_functions=config.fine_pos_encode)
    fine_model.to(device)
    fine_optimizer = torch.optim.Adam(fine_model.parameters(), lr=config.lr)

    # Train model.
    coarse_model.train()
    fine_model.train()

    train_sampler = RandomSampler(
        data_source=ray_dataset_train, num_samples=len(ray_dataset_train))
    dataloader = DataLoader(
        ray_dataset_train,
        sampler=train_sampler,
        batch_size=config.ray_batch_size,
    )

    writer = SummaryWriter(flush_secs=1)
    test_image_idx = 0
    test_image, _, _, _ = image_dataset_test[test_image_idx]
    writer.add_image("Test Image Ground Truth", test_image, dataformats="HWC")
    for it in tqdm(range(config.num_iters)):
        # One iteration of the training.
        for batch_i, (cam_orig_batch, cam_ray_batch, color_batch) in enumerate(tqdm(dataloader)):
            cam_orig_batch = cam_orig_batch.type(torch.float32).to(device)
            cam_ray_batch = cam_ray_batch.type(torch.float32).to(device)
            color_batch = color_batch.type(torch.float32).to(device)
            coarse_color_pred, occupancy, sample_depth = static_volumetric_renderer(
                coarse_model,
                cam_orig_batch.reshape(-1, 3),
                cam_ray_batch.reshape(-1, 3),
                config.nc,
                config.fc,
                num_sample=config.samples_per_ray,
                num_pos_encode=config.coarse_pos_encode,
                device=device
            )
            loss_coarse = torch.nn.functional.mse_loss(
                coarse_color_pred, color_batch)
            fine_color_pred, _, _ = adaptive_volumetric_renderer(
                fine_model,
                cam_orig_batch.reshape(-1, 3),
                cam_ray_batch.reshape(-1, 3),
                occupancy,
                sample_depth,
                max_subd_sample=config.max_subd_samples,
                num_pos_encode=config.fine_pos_encode,
                device=device
            )
            loss_fine = torch.nn.functional.mse_loss(
                fine_color_pred, color_batch)
            loss = loss_coarse + loss_fine
            loss.backward()
            coarse_optimizer.step()
            coarse_optimizer.zero_grad()
            fine_optimizer.step()
            fine_optimizer.zero_grad()
            writer.add_scalar("Coarse Loss", loss_coarse,
                              (it * len(dataloader) + batch_i) * config.ray_batch_size)
            writer.add_scalar("Fine Loss", loss_fine,
                              (it * len(dataloader) + batch_i) * config.ray_batch_size)
            if batch_i % config.save_image_every_n_batch == 0:
                coarse_model.eval()
                fine_model.eval()
                _, cam_transform, (h, w), (f, cx,
                                           cy) = image_dataset_test[test_image_idx]
                cam_orig, cam_ray_world = cam_ray_from_pose(
                    cam_transform, h, w, f, cx, cy)
                coarse_pred, fine_pred = nerf_image_render(
                    coarse_model,
                    fine_model,
                    cam_orig,
                    cam_ray_world,
                    config.ray_batch_size,
                    config.nc,
                    config.fc,
                    config.samples_per_ray,
                    config.max_subd_samples,
                    config.coarse_pos_encode,
                    config.fine_pos_encode,
                    device
                )
                writer.add_image("Rendered Test Image Fine", fine_pred, (it *
                                 len(dataloader) + batch_i) * config.ray_batch_size, dataformats="HWC")
                writer.add_image("Rendered Test Image Coarse", coarse_pred, (it *
                                 len(dataloader) + batch_i) * config.ray_batch_size, dataformats="HWC")
                coarse_model.train()
                fine_model.train()
        torch.save(coarse_model.state_dict(),
                   output_path / f"coarse_epoch_{it}.pth")
        torch.save(fine_model.state_dict(),
                   output_path / f"fine_epoch_{it}.pth")
    writer.close()

    torch.save(coarse_model.state_dict(),
               output_path / f"coarse_final.pth")
    torch.save(fine_model.state_dict(), output_path / f"fine_final.pth")


if __name__ == "__main__":
    main()
