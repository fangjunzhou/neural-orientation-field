import argparse
import pathlib
import logging
import sys
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from neural_orientation_field.nerf.dataset import NeRFImageDataset, NeRFRayDataset
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
        required=True,
        help="""
        Input images path.
        """,
        type=pathlib.Path
    )
    parser.add_argument(
        "-c",
        "--cam",
        required=True,
        help="""
        Input camera pose path.
        """,
        type=pathlib.Path
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="""
        Output model directory.
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
    image_dataset = NeRFImageDataset(
        frame_paths, cam_params, cam_transforms)
    num_train = len(image_dataset) - config.num_valid_image
    num_valid = config.num_valid_image
    image_dataset_train, image_dataset_valid = random_split(
        image_dataset, [num_train, num_valid])
    logging.info(f"Training image size: {len(image_dataset_train)}")
    logging.info(f"Validation image size: {len(image_dataset_valid)}")
    # Loading ray dataset.
    with tqdm(total=len(image_dataset_train), desc="Processing Image") as progress:
        ray_dataset_train = NeRFRayDataset(image_dataset_train, progress)

    # ------------------------- Training ------------------------- #

    # Save model parameters.

    model_params = {
        "coarse_pos_encode": config.coarse_pos_encode,
        "fine_pos_encode": config.fine_pos_encode,
        "nc": config.nc,
        "fc": config.fc,
        "samples_per_ray": config.samples_per_ray,
        "max_subd_samples": config.max_subd_samples,
    }
    torch.save(model_params, output_path / f"model_params.pth")

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
        data_source=ray_dataset_train, num_samples=int(len(ray_dataset_train)))
    dataloader = DataLoader(
        ray_dataset_train,
        sampler=train_sampler,
        batch_size=config.ray_batch_size,
    )

    writer = SummaryWriter(flush_secs=1)
    valid_images = np.array(
        [valid_image for valid_image, _, _, _ in image_dataset_valid]
    )
    writer.add_image("Valid Image Ground Truth",
                     valid_images, dataformats="NHWC")

    best_loss = float("inf")
    best_model_coarse = None
    best_model_fine = None

    for epoch in tqdm(range(config.num_epoch)):
        valid_every_n_batch = int(len(dataloader) / config.valid_per_epoch)
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
            coarse_loss = torch.nn.functional.mse_loss(
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
            fine_loss = torch.nn.functional.mse_loss(
                fine_color_pred, color_batch)
            loss = coarse_loss + fine_loss
            loss.backward()
            coarse_optimizer.step()
            coarse_optimizer.zero_grad()
            fine_optimizer.step()
            fine_optimizer.zero_grad()
            writer.add_scalar("Coarse Loss Train", coarse_loss,
                              (epoch * len(dataloader) + batch_i) * config.ray_batch_size)
            writer.add_scalar("Fine Loss Train", fine_loss,
                              (epoch * len(dataloader) + batch_i) * config.ray_batch_size)
            if batch_i % valid_every_n_batch == 0:
                coarse_model.eval()
                fine_model.eval()

                coarse_preds = []
                fine_preds = []
                coarse_losses = []
                fine_losses = []
                for valid_image, cam_transform, (h, w), (f, cx, cy) in image_dataset_valid:
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
                    coarse_preds.append(coarse_pred)
                    fine_preds.append(fine_pred)
                    valid_image = torch.tensor(valid_image)
                    coarse_loss = torch.nn.functional.mse_loss(
                        coarse_pred, valid_image)
                    fine_loss = torch.nn.functional.mse_loss(
                        fine_pred, valid_image)
                    coarse_losses.append(coarse_loss)
                    fine_losses.append(fine_loss)

                coarse_preds = np.array(coarse_preds)
                fine_preds = np.array(fine_preds)
                coarse_loss_valid = np.array(coarse_losses).mean()
                fine_loss_valid = np.array(fine_losses).mean()
                loss_valid = coarse_loss_valid + fine_loss_valid
                if loss_valid < best_loss:
                    best_model_coarse = coarse_model.state_dict()
                    best_model_fine = fine_model.state_dict()
                    best_loss = loss_valid
                writer.add_scalar("Coarse Loss Valid", coarse_loss_valid,
                                  (epoch * len(dataloader) + batch_i) * config.ray_batch_size)
                writer.add_scalar("Fine Loss Valid", fine_loss_valid,
                                  (epoch * len(dataloader) + batch_i) * config.ray_batch_size)
                writer.add_image("Rendered Validation Image Coarse", coarse_preds, (epoch *
                                 len(dataloader) + batch_i) * config.ray_batch_size, dataformats="NHWC")
                writer.add_image("Rendered Validation Image Fine", fine_preds, (epoch *
                                 len(dataloader) + batch_i) * config.ray_batch_size, dataformats="NHWC")
                coarse_model.train()
                fine_model.train()
        torch.save(coarse_model.state_dict(),
                   output_path / f"coarse_epoch_{epoch}.pth")
        torch.save(fine_model.state_dict(),
                   output_path / f"fine_epoch_{epoch}.pth")
    writer.close()

    if best_model_coarse:
        coarse_model.load_state_dict(best_model_coarse)
    if best_model_fine:
        fine_model.load_state_dict(best_model_fine)
    torch.save(coarse_model.state_dict(),
               output_path / f"coarse_final.pth")
    torch.save(fine_model.state_dict(), output_path / f"fine_final.pth")


if __name__ == "__main__":
    main()
