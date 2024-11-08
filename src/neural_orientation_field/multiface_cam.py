import argparse
import pathlib
import logging
import sys
import shutil


def main():
    parser = argparse.ArgumentParser(
        prog="Multiface Camera Extraction Helper",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="""
        The input path of the multiface image directory.
        """,
        type=pathlib.Path,
        required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        help="""
        The output path of the mutiface image for MVS.
        """,
        type=pathlib.Path,
        required=True
    )
    args = parser.parse_args()
    input_dir: pathlib.Path = args.input
    output_dir: pathlib.Path = args.output

    if not input_dir.exists():
        logging.error("The input directory doesn't exist.")
        sys.exit(1)
    if not output_dir.exists():
        logging.error("The output directory doesn't exist.")
        sys.exit(1)
    if any(output_dir.iterdir()):
        logging.error("The output directory is not empty.")
        sys.exit(1)

    frames: dict[str, list[pathlib.Path]] = dict()
    for cam_dir in input_dir.glob("*"):
        cam_name = cam_dir.stem
        for image_file in cam_dir.glob("*"):
            frame_name = image_file.stem
            if not frame_name in frames:
                frames[frame_name] = []
            frames[frame_name].append(image_file)

    for frame_name, images in frames.items():
        frame_dir = output_dir / frame_name
        frame_dir.mkdir()
        for idx, image_src_path in enumerate(images):
            image_dst_path = frame_dir / f"{idx}.png"
            logging.info(f"Copying from {image_src_path} to {image_dst_path}.")
            shutil.copy(image_src_path, image_dst_path)


if __name__ == "__main__":
    main()
