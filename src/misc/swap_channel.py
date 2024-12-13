import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def process_images(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Check if the file is an image
        try:
            image = Image.open(input_path)
        except IOError:
            print(f"Skipping non-image file: {filename}")
            continue

        image_array = np.array(image)

        new_img_array = np.zeros_like(image_array)
        new_img_array[:, :, 0] = image_array[:, :, 1]  # Swap Green to Red
        new_img_array[:, :, 1] = image_array[:, :, 2]  # Swap Blue to Green
        # new_img_array[:, :, 2] = image_array[:, :, 0]  # Swap Red to Blue

        # Convert back to an image
        new_image = Image.fromarray(new_img_array)

        # Save the processed image to the output folder
        output_path = os.path.join(output_folder, filename)
        new_image.save(output_path)

        print(f"Processed and saved: {output_path}")


# Example usage
if __name__ == "__main__":
    input_folder = "groundtruth_hairstep"
    output_folder = "groundtruth_hairstep_converted"

    process_images(input_folder, output_folder)
