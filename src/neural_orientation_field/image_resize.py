import argparse
import pathlib
import logging
import sys

from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        prog="Multiface Camera Extraction Helper",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="""
        The input image directory.
        """,
        type=pathlib.Path,
        required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        help="""
        The output image directory.
        """,
        type=pathlib.Path,
        required=True
    )
    parser.add_argument(
        "-r",
        "--rate",
        default=0.5,
        help="""
        Down sample rate.
        """,
        type=float
    )
    args = parser.parse_args()
    input_dir: pathlib.Path = args.input
    output_dir: pathlib.Path = args.output
    sample_rate: float = args.rate

    if not input_dir.exists():
        logging.error("The input directory doesn't exist.")
        sys.exit(1)
    if not output_dir.exists():
        logging.error("The output directory doesn't exist.")
        sys.exit(1)

    for image_file_path in input_dir.glob("*"):
        try:
            image = Image.open(image_file_path)
        except Exception as e:
            continue
        new_width = int(image.width * sample_rate)
        new_height = int(image.height * sample_rate)
        image = image.resize((new_width, new_height))
        image.save(output_dir / image_file_path.name)


if __name__ == "__main__":
    main()
