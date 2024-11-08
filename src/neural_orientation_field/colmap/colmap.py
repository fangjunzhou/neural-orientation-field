import json
import pathlib
import argparse
import logging
import sys

import numpy as np

import pycolmap
import pyvista as pv

import neural_orientation_field.utils as utils


def main():
    # ---------------------- Argument Setup ---------------------- #
    parser = argparse.ArgumentParser(
        prog="COLMAP Preprocessor",
        description="""
        Extract point cloud and camera position using COLMAP. 
        """
    )
    parser.add_argument(
        "-p",
        "--path",
        default=pathlib.Path("./nof-config.json"),
        help="""
        The project config json file path. The default path is 
        ./nof-config.json
        """,
        type=pathlib.Path
    )
    # ------------------- Read Project Config  ------------------- #
    args = parser.parse_args()
    config_path: pathlib.Path = args.path
    if not config_path.exists():
        logging.error("The project config doesn't exist.")
        sys.exit(1)
    with open(config_path, "r") as config_file:
        try:
            config_dict = json.load(config_file)
            project_config = utils.ProjectConfig.from_dict(config_dict)
        except Exception as e:
            logging.error("Failed to load project config.", stack_info=True)
            sys.exit(1)
    # --------------------- COLMAP Pipeline  --------------------- #
    # Data path.
    image_dir = project_config.input_path
    db_path = project_config.cache_path / "colmap-db.db"
    output_dir = project_config.cache_path / "colmap"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    # COLMAP pipeline.
    pycolmap.extract_features(
        db_path, image_dir, camera_model="SIMPLE_PINHOLE")
    pycolmap.match_exhaustive(db_path)
    maps = pycolmap.incremental_mapping(db_path, image_dir, output_dir)


if __name__ == "__main__":
    main()
