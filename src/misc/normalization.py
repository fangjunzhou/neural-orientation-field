import numpy as np
from PIL import Image
import math
import os
import matplotlib.pyplot as plt

def normalize_dir(img_arr, x_idx=0, y_idx=1, out_x_idx=0, out_y_idx=1):
    x, y = img_arr[:, :, x_idx], img_arr[:, :, y_idx]
    x_norm, y_norm = x / np.sqrt(x**2 + y**2),  y / np.sqrt(x**2 + y**2)
    output = np.stack([x_norm, y_norm], axis=-1)
    return output

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                img = Image.open(file_path).convert("RGB")
                images.append(np.array(img))
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    return images

if __name__ == '__main__':
    image_path = 'src\\scripts\\color_ring.jpg'

    gthsc = load_images_from_folder('src\\scripts\\groundtruth_hairstep_converted')
    nerfhsc = load_images_from_folder('src\\scripts\\nerf_hairstep_converted')

    gt = load_images_from_folder('src\\scripts\\groundtruth')
    neof = load_images_from_folder('src\\scripts\\neof')


    for idx, img in enumerate(nerfhsc):

        norm_xy = normalize_dir(img.astype(np.float64))
        
        outxy = np.zeros_like(img)
        outxy[:, :, :2] = norm_xy * 255
        plt.imshow(outxy)
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.savefig(f"output_test_{idx}.png", bbox_inches='tight', pad_inches=0, transparent=True)
        print(f"{idx}done")
        