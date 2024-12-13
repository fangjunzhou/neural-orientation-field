import os
# import cv2
from PIL import Image
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):  # Sort for consistent pairing
        filepath = os.path.join(folder, filename)
        image = Image.open(filepath)
        images.append(np.array(image.convert("RGB")).astype(np.float32))  # Convert to float for MSE computation
    return images

def calculate_mse_psnr(images, ground_truths):

    total_mse = 0
    num_images = len(images)
    max_pixel = 255.0  # Assuming 8-bit images

    for img, gt in zip(images, ground_truths):
        if img.shape != gt.shape:
            raise ValueError("Each image and its ground truth must have the same dimensions.")
        
        # Calculate MSE_k for this image
        mse_k = np.mean((img[:, :, :2] - gt[:, :, :2]) ** 2)
        total_mse += mse_k

    # Calculate average MSE
    mse_avg = total_mse / num_images

    # Calculate PSNR_avg
    if mse_avg == 0:
        psnr_avg = float('inf')  # No error, PSNR is infinite
    else:
        psnr_avg = 10 * np.log10(max_pixel**2 / mse_avg)

    return mse_avg, psnr_avg


if __name__ == "__main__":
    # Paths to folders containing reconstructed images and ground truth images
    ground_truth_folder = "src\\scripts\\paper_gt"

    folders = ['src\\scripts\\paper_neof',
               'src\\scripts\\paper_hs',
               'src\\scripts\\paper_nerf']

    # Load images from the folders
    ground_truth_images = load_images_from_folder(ground_truth_folder)

    for folder in folders:
        reconstructed_images = load_images_from_folder(folder)

        mse_avg, psnr_avg = calculate_mse_psnr(reconstructed_images, ground_truth_images)

        print(f"{folder}")

        print(f"Average MSE: {mse_avg:.6f}")
        print(f"Average PSNR: {psnr_avg:.2f} dB")
