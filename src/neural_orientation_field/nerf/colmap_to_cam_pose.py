import argparse
import pathlib
import logging
import sys

import numpy as np
import pycolmap

import neural_orientation_field.colmap.colmap_utils as colutils


def main():
    # ---------------------- Argument Setup ---------------------- #

    parser = argparse.ArgumentParser(
        prog="COLMAP to Camera Pose",
        description="""
        COLMAP camera pose extractor.
        """
    )
    parser.add_argument(
        "-c",
        "--colmap",
        required=True,
        help="""
        Input COLMAP model path.
        """,
        type=pathlib.Path
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="""
        Output camera pose directory.
        """,
        type=pathlib.Path
    )

    args = parser.parse_args()
    colmap_model_path: pathlib.Path = args.colmap
    if not colmap_model_path.exists():
        logging.error("The input COLMAP model directory doesn't exist.")
        sys.exit(1)
    output_path: pathlib.Path = args.output
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # ------------------- Extract Camera Pose  ------------------- #

    # Load COLMAP reconstruction.
    colmap_model = pycolmap.Reconstruction(colmap_model_path)
    # NeRF requires same camera.
    if colmap_model.num_cameras() != 1:
        raise ValueError(
            "Only COMAP reconstructions with single camera is accepted.")
    cam_transforms, cam_params, image_file_names = colutils.get_camera_poses(
        colmap_model)
    f, cx, cy = cam_params[0]
    for i in range(cam_transforms.shape[0]):
        cam_transforms[i] = np.linalg.inv(cam_transforms[i])

    with open(output_path / "frame-names.txt", "w") as frame_name_file:
        for image_file_name in image_file_names[:-1]:
            frame_name_file.writelines(image_file_name + "\n")
        frame_name_file.writelines(image_file_names[-1])

    with open(output_path / "camera-transforms.npy", "wb") as cam_transform_file:
        np.save(cam_transform_file, cam_transforms)

    with open(output_path / "camera-params.npy", "wb") as cam_params_file:
        np.save(cam_params_file, (f, cx, cy))


if __name__ == "__main__":
    main()
