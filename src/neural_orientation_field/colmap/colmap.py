import pathlib
import argparse
import logging
import sys

import pycolmap


def main():
    # ---------------------- Argument Setup ---------------------- #
    parser = argparse.ArgumentParser(
        prog="COLMAP Preprocessor",
        description="""
        Extract point cloud and camera position using COLMAP. 
        """
    )
    parser.add_argument(
        "-i",
        "--input",
        default=pathlib.Path("./data/images/"),
        help="""
        Input images directory.
        """,
        type=pathlib.Path
    )
    parser.add_argument(
        "-o",
        "--output",
        default=pathlib.Path("./data/output/colmap/"),
        help="""
        Output model directory.
        """,
        type=pathlib.Path
    )
    # ------------------- Read Project Config  ------------------- #
    args = parser.parse_args()
    input_path: pathlib.Path = args.input
    if not input_path.exists():
        logging.error("The input image directory doesn't exist.")
        sys.exit(1)
    output_path: pathlib.Path = args.output
    if not output_path.exists():
        output_path.mkdir(parents=True)
    # --------------------- COLMAP Pipeline  --------------------- #
    # Data path.
    image_dir = input_path
    db_path = output_path / "colmap-db.db"
    output_dir = output_path / "model"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    # COLMAP pipeline.
    pycolmap.extract_features(
        db_path, input_path, camera_mode=pycolmap.CameraMode.SINGLE, camera_model="SIMPLE_PINHOLE")
    pycolmap.match_exhaustive(db_path)
    maps = pycolmap.incremental_mapping(db_path, input_path, output_dir)


if __name__ == "__main__":
    main()
