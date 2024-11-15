import os
import argparse
from PIL import Image
import numpy as np

def compute_mse_between_folders(ground_truth_folder, generated_folder):
    ground_truth_images = sorted([os.path.join(ground_truth_folder, img) for img in os.listdir(ground_truth_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
    generated_images = sorted([os.path.join(generated_folder, img) for img in os.listdir(generated_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])

    if len(ground_truth_images) != len(generated_images):
        raise ValueError("Number of images in the two folders are not equal!")

    total_mse = 0
    

    for gt_img_path, gen_img_path in zip(ground_truth_images, generated_images):
        gt_img = Image.open(gt_img_path).convert('RGB')
        gen_img = Image.open(gen_img_path).convert('RGB')

        # Convert PIL Image to numpy array
        gt_array = np.array(gt_img, dtype=np.float32)
        gen_array = np.array(gen_img, dtype=np.float32)

        # Compute MSE using numpy
        mse = np.mean((gt_array - gen_array) ** 2)
        total_mse += mse

    average_mse = total_mse / len(ground_truth_images)
    return average_mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MSE between two folders of images.")
    parser.add_argument("--gt_folder", type=str, required=True, help="Path to the ground truth images folder")
    parser.add_argument("--gen_folder", type=str, required=True, help="Path to the generated images folder")
    args = parser.parse_args()

    mse = compute_mse_between_folders(args.gt_folder, args.gen_folder)
    print(f"Average MSE: {mse}")
