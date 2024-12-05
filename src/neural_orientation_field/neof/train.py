import argparse
import pathlib
import logging
import sys
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from neural_orientation_field.neof.model import NeOFCoarseModel, NeOFFineModel
from neural_orientation_field.neof.dataset import NeOFImageDataset, NeOFRayDataset
from neural_orientation_field.neof.training_config import NeOFTrainingConfig
from neural_orientation_field.neof.utils import hair_dir_vec2color, static_volumetric_renderer, adaptive_volumetric_renderer, nerf_image_render
from neural_orientation_field.nerf.utils import cam_ray_from_pose


def main():
    logging.basicConfig(level=logging.INFO)
    # ---------------------- Argument Setup ---------------------- #
    parser = argparse.ArgumentParser(
        prog="NeOF Trainer",
        description="""
        NeOF trainer.
        """
    )
    parser.add_argument(
        "-bm",
        "--body_mask",
        required=True,
        help="""
        Body mask path.
        """,
        type=pathlib.Path
    )
    parser.add_argument(
        "-hm",
        "--hair_mask",
        required=True,
        help="""
        Hair mask path.
        """,
        type=pathlib.Path
    )
    parser.add_argument(
        "-hd",
        "--hair_dir",
        required=True,
        help="""
        Hair direction path.
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
    body_mask_path: pathlib.Path = args.body_mask
    if not body_mask_path.exists():
        logging.error("The body mask directory doesn't exist.")
        sys.exit(1)
    hair_mask_path: pathlib.Path = args.hair_mask
    if not hair_mask_path.exists():
        logging.error("The hair mask directory doesn't exist.")
        sys.exit(1)
    hair_dir_path: pathlib.Path = args.hair_dir
    if not hair_dir_path.exists():
        logging.error("The hair direction directory doesn't exist.")
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

    config = NeOFTrainingConfig()

    # ----------------------- Load Dataset ----------------------- #

    # Load camera parameters.
    frame_name_path = cam_path / "frame-names.txt"
    cam_transform_path = cam_path / "camera-transforms.npy"
    cam_param_path = cam_path / "camera-params.npy"
    with open(frame_name_path, "r") as frame_path_file:
        frame_names = frame_path_file.read().split("\n")
        body_mask_paths = [body_mask_path /
                           frame_name for frame_name in frame_names]
        hair_mask_paths = [hair_mask_path /
                           frame_name for frame_name in frame_names]
        hair_dir_paths = [hair_dir_path /
                          frame_name for frame_name in frame_names]
    with open(cam_transform_path, "rb") as cam_transform_file:
        cam_transforms = np.load(cam_transform_file)
    with open(cam_param_path, "rb") as cam_param_file:
        cam_params = np.load(cam_param_file)

    # Load image dataset.
    image_dataset = NeOFImageDataset(
        body_mask_paths, hair_mask_paths, hair_dir_paths, cam_params, cam_transforms)
    num_train = len(image_dataset) - config.num_valid_image
    num_valid = config.num_valid_image
    image_dataset_train, image_dataset_valid = random_split(
        image_dataset, [num_train, num_valid])
    logging.info(f"Training image size: {len(image_dataset_train)}")
    logging.info(f"Validation image size: {len(image_dataset_valid)}")
    # Loading ray dataset.
    with tqdm(total=len(image_dataset_train), desc="Processing Image") as progress:
        ray_dataset_train = NeOFRayDataset(image_dataset_train, progress)

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
    coarse_model = NeOFCoarseModel(
        num_encoding_functions=config.coarse_pos_encode)
    coarse_model.to(device)
    coarse_optimizer = torch.optim.Adam(
        coarse_model.parameters(), lr=config.lr)

    fine_model = NeOFFineModel(num_encoding_functions=config.fine_pos_encode)
    fine_model.to(device)
    fine_optimizer = torch.optim.Adam(fine_model.parameters(), lr=config.lr)

    coarse_model.train()
    fine_model.train()

    train_sampler = RandomSampler(data_source=ray_dataset_train, num_samples=int(
        config.size_train_ray * len(ray_dataset_train)))
    dataloader = DataLoader(
        ray_dataset_train,
        sampler=train_sampler,
        batch_size=config.ray_batch_size,
    )

    writer = SummaryWriter(flush_secs=1)
    valid_hair_dirs = []
    for (_, _, valid_hair_dir), _, _, _ in image_dataset_valid:
        valid_hair_dirs.append(hair_dir_vec2color(valid_hair_dir))
    valid_hair_dirs = np.array(valid_hair_dirs)
    writer.add_image("Valid Hair Direction Ground Truth",
                     valid_hair_dirs, dataformats="NHWC")

    best_loss = float("inf")
    best_model_coarse = None
    best_model_fine = None

    for epoch in tqdm(range(config.num_epoch)):
        valid_every_n_batch = int(len(dataloader) / config.valid_per_epoch)
        # One iteration of the training.
        for batch_i, (
            cam_trans_batch,
            cam_orig_batch,
            cam_ray_batch,
            body_mask_batch,
            hair_mask_batch,
            hair_dir_batch
        ) in enumerate(tqdm(dataloader)):
            cam_trans_batch = cam_trans_batch.type(torch.float32).to(device)
            cam_orig_batch = cam_orig_batch.type(torch.float32).to(device)
            cam_ray_batch = cam_ray_batch.type(torch.float32).to(device)
            body_mask_batch = body_mask_batch.type(torch.float32).to(device)
            hair_mask_batch = hair_mask_batch.type(torch.float32).to(device)
            hair_dir_batch = hair_dir_batch.type(torch.float32).to(device)
            coarse_ss_orientation, coarse_occupancy, sample_depths = static_volumetric_renderer(
                coarse_model,
                cam_trans_batch.reshape(-1, 4, 4),
                cam_orig_batch.reshape(-1, 3),
                cam_ray_batch.reshape(-1, 3),
                config.nc,
                config.fc,
                num_sample=config.samples_per_ray,
                num_pos_encode=config.coarse_pos_encode,
                device=device
            )
            sample_depths_diff = sample_depths[:, 1:] - sample_depths[:, :-1]
            occupancy_hair = coarse_occupancy[:, :, 0]
            occupancy_body = 1 - coarse_occupancy[:, :, 2]
            occupancy_hair = occupancy_hair * sample_depths_diff
            coarse_hair_mask = 1 - torch.exp(-occupancy_hair.sum(dim=-1))
            coarse_hair_mask_loss = torch.nn.functional.mse_loss(
                coarse_hair_mask, hair_mask_batch.flatten())
            occupancy_body = occupancy_body * sample_depths_diff
            coarse_body_mask = 1 - torch.exp(-occupancy_body.sum(dim=-1))
            coarse_body_mask_loss = torch.nn.functional.mse_loss(
                coarse_body_mask, body_mask_batch.flatten())
            coarse_ss_orientation_loss = torch.nn.functional.mse_loss(
                coarse_ss_orientation, hair_dir_batch)
            coarse_loss = coarse_ss_orientation_loss + \
                coarse_hair_mask_loss + coarse_body_mask_loss

            fine_ss_orientation, fine_occupancy, sample_depths = adaptive_volumetric_renderer(
                fine_model,
                cam_trans_batch.reshape(-1, 4, 4),
                cam_orig_batch.reshape(-1, 3),
                cam_ray_batch.reshape(-1, 3),
                coarse_occupancy[:, :, 0] + coarse_occupancy[:, :, 1],
                sample_depths,
                max_subd_sample=config.max_subd_samples,
                num_pos_encode=config.fine_pos_encode,
                device=device
            )
            sample_depths_diff = sample_depths[:, 1:] - sample_depths[:, :-1]
            occupancy_hair = fine_occupancy[:, :, 0]
            occupancy_body = 1 - fine_occupancy[:, :, 2]
            occupancy_hair = occupancy_hair * sample_depths_diff
            fine_hair_mask = 1 - torch.exp(-occupancy_hair.sum(dim=-1))
            fine_hair_mask_loss = torch.nn.functional.mse_loss(
                fine_hair_mask, hair_mask_batch.flatten())
            occupancy_body = occupancy_body * sample_depths_diff
            fine_body_mask = 1 - torch.exp(-occupancy_body.sum(dim=-1))
            fine_body_mask_loss = torch.nn.functional.mse_loss(
                fine_body_mask, body_mask_batch.flatten())
            fine_ss_orientation_loss = torch.nn.functional.mse_loss(
                fine_ss_orientation, hair_dir_batch)
            fine_loss = fine_ss_orientation_loss + fine_hair_mask_loss + fine_body_mask_loss

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
                for (_, _, valid_hair_dir), cam_transform, (h, w), (f, cx, cy) in image_dataset_valid:
                    cam_orig, cam_ray_world = cam_ray_from_pose(
                        cam_transform, h, w, f, cx, cy)
                    coarse_pred, fine_pred = nerf_image_render(
                        coarse_model,
                        fine_model,
                        cam_transform,
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
                    coarse_preds.append(
                        hair_dir_vec2color(coarse_pred.numpy()))
                    fine_preds.append(hair_dir_vec2color(fine_pred.numpy()))
                    valid_hair_dir = torch.tensor(valid_hair_dir)
                    coarse_loss = torch.nn.functional.mse_loss(
                        coarse_pred, valid_hair_dir)
                    fine_loss = torch.nn.functional.mse_loss(
                        fine_pred, valid_hair_dir)
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

        torch.save(coarse_model.state_dict(), output_path /
                   f"coarse_epoch_{epoch}.pth")
        torch.save(fine_model.state_dict(), output_path /
                   f"fine_epoch_{epoch}.pth")
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
