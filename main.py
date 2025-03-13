import cv2
import numpy as np
import glob
import os
import imageio
import psutil
import sys
import argparse
import concurrent.futures
from tqdm import tqdm

def set_low_priority():
    """Sets the process priority to low (Windows-compatible)."""
    try:
        p = psutil.Process(os.getpid())
        if sys.platform == "win32":
            p.nice(psutil.IDLE_PRIORITY_CLASS)  # Windows low priority
        else:
            os.nice(19)  # Linux/Mac
    except Exception as e:
        print(f"Warning: Could not set process priority: {e}")

def align_image(img, ref_image, keypoints_ref, descriptors_ref, orb, flann):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints_img, descriptors_img = orb.detectAndCompute(gray_img, None)
    
    if descriptors_img is None or descriptors_ref is None:
        print("Warning: No keypoints found, image will not be aligned.")
        return img
    
    matches = flann.knnMatch(descriptors_img, descriptors_ref, k=2)
    good_matches = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.75 * n.distance]
    
    if len(good_matches) > 10:
        src_pts = np.float32([keypoints_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return cv2.warpPerspective(img, matrix, (ref_image.shape[1], ref_image.shape[0]), flags=cv2.INTER_CUBIC)
    
    print("Warning: Too few matches found, image will not be aligned.")
    return img

def align_images_orb(images):
    ref_image = images[0]
    gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=500)
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    keypoints_ref, descriptors_ref = orb.detectAndCompute(gray_ref, None)
    aligned_images = [ref_image]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(align_image, img, ref_image, keypoints_ref, descriptors_ref, orb, flann) for img in images[1:]]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Aligning images", unit="image"):
            aligned_images.append(future.result())
    
    return aligned_images

def remove_moving_objects(image_folder, output_filename):
    set_low_priority()
    
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    images = [cv2.imread(img) for img in image_paths]
    
    if len(images) < 2:
        print("Not enough images found for processing.")
        return
    
    print("Aligning images...")
    aligned_images = align_images_orb(images)
    
    print("Computing background...")
    channels = cv2.split(np.array(aligned_images))
    median_channels = [np.median(ch, axis=0).astype(np.uint8) for ch in channels]
    median_image = cv2.merge(median_channels)
    
    # Convert BGR to RGB
    median_image_rgb = cv2.cvtColor(median_image, cv2.COLOR_BGR2RGB)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save with imageio
    try:
        imageio.imwrite(output_filename, median_image_rgb)
        print(f"Background image saved as {output_filename}")
    except Exception as e:
        print(f"Error saving {output_filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Removes moving objects from a series of images by extracting a static background.",
        epilog="Example: python script.py /path/to/images output.jpg"
    )
    parser.add_argument("image_folder", type=str, help="Path to the folder containing images (must exist and contain .jpg files).")
    parser.add_argument("output_filename", type=str, help="Path and filename for the output image (e.g., 'background.jpg').")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.image_folder):
        print("Error: The specified image folder does not exist or is invalid.")
        parser.print_help()
        sys.exit(1)
    
    if not args.output_filename.lower().endswith(".jpg"):
        print("Error: The output file must be a .jpg file.")
        parser.print_help()
        sys.exit(1)
    
    remove_moving_objects(args.image_folder, args.output_filename)
