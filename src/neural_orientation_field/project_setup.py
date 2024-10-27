import logging
import argparse
import shutil
import pathlib
import json
import sys

import neural_orientation_field.utils as utils

logging.basicConfig(level=logging.DEBUG)

def main():
    # ---------------------- Argument Setup ---------------------- #
    parser = argparse.ArgumentParser(
        prog="Neural Orientation Field Project Setup Helper",
        description="""
        Automatically config the data and cache directory for COLMAP 
        preprocessor and Neural Orientation Field trainer.
        """
    )
    parser.add_argument(
        "-i",
        "--input",
        default=pathlib.Path("./data/images/"),
        help="""
        The input path for the entire pipeline. Should be a directory
        containing all the images to be processed. The default path is
        ./data/images/
        """,
        type=pathlib.Path
    )
    parser.add_argument(
        "-o",
        "--output",
        default=pathlib.Path("./data/output/"),
        help="""
        The output path for the training. Should be a directory to store the
        trained model. The default path is ./data/output/
        """,
        type=pathlib.Path
    )
    parser.add_argument(
        "-c",
        "--cache",
        default=pathlib.Path("./data/cache/"),
        help="""
        The cache path for the entire pipeline. Should be a directory storing
        all the intermediate cache including COLMAP database, etc. The default
        path is ./data/cache/
        """,
        type=pathlib.Path
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
    parser.add_argument(
        "--clear",
        action="store_true",
        help="""
        If clearing all the existing cache and output directory.
        """
    )
    # --------------------- Argument Parsing --------------------- #
    args = parser.parse_args()
    project_config = utils.ProjectConfig(
        input_path=args.input,
        output_path=args.output,
        cache_path=args.cache
    )
    config_path: pathlib.Path = args.path
    clear: bool = args.clear
    # ---------------------- Project Setup  ---------------------- #
    # Check input directory.
    if not project_config.input_path.exists():
        logging.error("The input directory doesn't exist.")
        sys.exit(1)
    if not project_config.input_path.is_dir():
        logging.error("The input path is not a directory. See --help for " +
                      "more details")
        sys.exit(1)
    # Setup output directory.
    if project_config.output_path.exists():
        if not project_config.output_path.is_dir():
            logging.error("The output path is not a directory. See --help " +
                          "for more details.")
            sys.exit(1)
        if clear:
            print("The project setup script will remove " + 
                  f"{str(project_config.output_path.resolve())}, " +
                  "are you sure to continue?")
            if not utils.confirm():
                sys.exit(0)
            print(f"Removing {str(project_config.output_path.resolve())}")
            shutil.rmtree(project_config.output_path)
            project_config.output_path.mkdir(parents=True)
    else:
        project_config.output_path.mkdir(parents=True)
    # Setup cache directory.
    if project_config.cache_path.exists():
        if not project_config.cache_path.is_dir():
            logging.error("The cache path is not a directory. See --help " +
                          "for more details.")
            sys.exit(1)
        if clear:
            print("The project setup script will remove " + 
                  f"{str(project_config.cache_path.resolve())}, " +
                  "are you sure to continue?")
            if not utils.confirm():
                sys.exit(0)
            print(f"Removing {str(project_config.cache_path.resolve())}")
            shutil.rmtree(project_config.cache_path)
            project_config.cache_path.mkdir(parents=True)
    else:
        project_config.cache_path.mkdir(parents=True)
    # Save the config.
    with open(config_path, "w") as config_file:
        json.dump(
            project_config.to_dict(),
            config_file,
            indent=2
        )
    print(f"Project config saved to {str(config_path.resolve())}")

if __name__ == "__main__":
    main()
