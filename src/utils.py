# src/utils.py - Minimalistic Version

import cv2
import numpy as np
import os
import random 

from src.logger import logging


def load_image_mask_pair(image_path, mask_path, target_size=(512,512)):
    """Loads & resizes image (RGB) and mask (Gray)."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    logging.info(f"Resized image and mask to {target_size}")

    return resized_image, resized_mask # image: HxWx3 uint8, mask: HxW uint8 (0-255)


def basic_normalize_and_binarize(image_np, mask_np, mask_split_threshold=128):
    """Normalizes image (0-1) & binarizes mask (0 or 1)."""
    # Normalize image (0-255 to 0-1)
    normalized_img_np = image_np.astype(np.float32) / 255.0

    # Binarize mask (assuming values > threshold are water)
    binarized_mask_np = (mask_np > mask_split_threshold).astype(np.uint8) # Now values are 0 or 1

    logging.info(f"Normalizing image and binarizing mask with threshold {mask_split_threshold}")
    return normalized_img_np, binarized_mask_np # image: HxWx3 float32, mask: HxW uint8 (0/1)


def save_processed_pair(image_np, mask_np, processed_image_path, processed_mask_path):
    """Saves processed image (RGB float 0-1) & binarized mask (Gray uint8 0/1)."""
    os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_mask_path), exist_ok=True)

    logging.info(f"Saving processed image to: {processed_image_path}")
    logging.info(f"Saving processed mask to: {processed_mask_path}")

    # Convert normalized float [0, 1] image back to uint8 [0, 255] and RGB->BGR for saving
    image_save_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_save_np, cv2.COLOR_RGB2BGR)

    # Convert binarized mask [0, 1] to [0, 255] for saving as grayscale image
    mask_save_np = (mask_np * 255).astype(np.uint8)

    cv2.imwrite(processed_image_path, image_bgr)
    cv2.imwrite(processed_mask_path, mask_save_np)

    logging.info("Image and mask saved successfully.")

# --- Add placeholder functions for Data Augmentation and Tiling here ---
# You can leave them empty or add minimal logic if you get time

def data_augment(image_np, mask_np):
    """Applies data augmentation (placeholder)."""
    logging.info("Applying data augmentation...")

    if random.random() > 0.5: # 50% chance
        image_np = np.fliplr(image_np)
        mask_np = np.fliplr(mask_np)

    # --- Random Vertical Flip ---
    if random.random() > 0.5: # 50% chance
        image_np = np.flipud(image_np)
        mask_np = np.flipud(mask_np)

    return image_np, mask_np

def tile_image_mask(image, mask, tile_size=(256, 256), overlap=0):
    """
    Tiles large image and mask into smaller patches with optional overlap.
    
    Parameters:
    - image (np.ndarray): RGB image (H x W x 3)
    - mask (np.ndarray): Binary mask (H x W)
    - tile_size (tuple): Desired tile size (height, width)
    - overlap (int): Number of pixels to overlap between tiles
    
    Returns:
    - List of (image_tile, mask_tile) tuples
    """

    logging.info(f"Tiling image and mask with tile size {tile_size} and overlap {overlap}")

    tile_h, tile_w = tile_size
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    h, w = image.shape[:2]
    
    image_tiles = []
    for y in range(0, h - tile_h + 1, stride_h):
        for x in range(0, w - tile_w + 1, stride_w):
            img_tile = image[y:y+tile_h, x:x+tile_w]
            mask_tile = mask[y:y+tile_h, x:x+tile_w]
            image_tiles.append((img_tile, mask_tile))

    # Handle edge tiles (if image size not divisible by tile size)
    if h % stride_h != 0:
        for x in range(0, w - tile_w + 1, stride_w):
            img_tile = image[h - tile_h:h, x:x + tile_w]
            mask_tile = mask[h - tile_h:h, x:x + tile_w]
            image_tiles.append((img_tile, mask_tile))

    if w % stride_w != 0:
        for y in range(0, h - tile_h + 1, stride_h):
            img_tile = image[y:y + tile_h, w - tile_w:w]
            mask_tile = mask[y:y + tile_h, w - tile_w:w]
            image_tiles.append((img_tile, mask_tile))

    # Bottom-right corner
    if h % stride_h != 0 and w % stride_w != 0:
        img_tile = image[h - tile_h:h, w - tile_w:w]
        mask_tile = mask[h - tile_h:h, w - tile_w:w]
        image_tiles.append((img_tile, mask_tile))

    logging.info(f"Tiling complete: {len(image_tiles)} tiles created.")
    return image_tiles
