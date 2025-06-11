import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import (
    load_image_mask_pair,
    basic_normalize_and_binarize,
    save_processed_pair,
    data_augment,
    tile_image_mask
)

# --- Configuration ---
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
NUM_SAMPLES = len(os.listdir(os.path.join(RAW_DATA_DIR, 'images')))  # Process all images by default
TARGET_SIZE = (512, 512)
MASK_THRESHOLD = 1
NUM_AUGMENTATIONS_PER_SAMPLE = 2
TILE_SIZE = (128,128)
TILE_OVERLAP = 0
ENABLE_TILING = True  


def run_ingestion_pipeline(raw_dir, processed_dir, num_samples=None, target_size=TARGET_SIZE, mask_threshold=1, num_augmentations=1):
    print(f"Starting ingestion and preprocessing...")
    os.makedirs(os.path.join(processed_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, 'masks'), exist_ok=True)

    raw_image_dir = os.path.join(raw_dir, 'images')
    raw_mask_dir = os.path.join(raw_dir, 'masks')

    image_files = [f for f in os.listdir(raw_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    if num_samples is not None:
        image_files = image_files[:num_samples]

    print(f"Found {len(image_files)} original files to process.")

    processed_count = 0
    skipped_count = 0

    for img_file in image_files:
        img_path = os.path.join(raw_image_dir, img_file)
        mask_path = os.path.join(raw_mask_dir, img_file)

        if not os.path.exists(mask_path):
            print(f"Skipping {img_file}: Mask not found at {mask_path}")
            skipped_count += 1
            continue

        try:
            img_np, mask_np = load_image_mask_pair(img_path, mask_path, target_size=target_size)
            processed_img, processed_mask = basic_normalize_and_binarize(img_np, mask_np, mask_split_threshold=mask_threshold)
            base_name = os.path.splitext(img_file)[0]

            # --- Process original image ---
            if ENABLE_TILING:
                tiles = tile_image_mask(processed_img, processed_mask, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)
                for t_idx, (img_tile, mask_tile) in enumerate(tiles):
                    img_name = f"{base_name}_orig_tile{t_idx}.png"
                    mask_name = f"{base_name}_orig_tile{t_idx}.png"
                    save_processed_pair(img_tile, mask_tile,
                                        os.path.join(processed_dir, 'images', img_name),
                                        os.path.join(processed_dir, 'masks', mask_name))
                    processed_count += 1
                # print(f"Saved {len(tiles)} tiles from original: {base_name}")
            else:
                img_name = f"{base_name}_orig.png"
                save_processed_pair(processed_img, processed_mask,
                                    os.path.join(processed_dir, 'images', img_name),
                                    os.path.join(processed_dir, 'masks', img_name))
                processed_count += 1
                print(f"Saved original: {img_name}")

            # --- Process augmentations ---
            for i in range(num_augmentations - 1):
                augmented_img, augmented_mask = data_augment(processed_img.copy(), processed_mask.copy())
                if ENABLE_TILING:
                    tiles = tile_image_mask(augmented_img, augmented_mask, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)
                    for t_idx, (img_tile, mask_tile) in enumerate(tiles):
                        img_name = f"{base_name}_aug{i+1}_tile{t_idx}.png"
                        mask_name = f"{base_name}_aug{i+1}_tile{t_idx}.png"
                        save_processed_pair(img_tile, mask_tile,
                                            os.path.join(processed_dir, 'images', img_name),
                                            os.path.join(processed_dir, 'masks', mask_name))
                        processed_count += 1
                    # print(f"Saved {len(tiles)} tiles from augmentation {i+1}: {base_name}")
                else:
                    img_name = f"{base_name}_aug{i+1}.png"
                    save_processed_pair(augmented_img, augmented_mask,
                                        os.path.join(processed_dir, 'images', img_name),
                                        os.path.join(processed_dir, 'masks', img_name))
                    processed_count += 1
                    print(f"Saved augmented version {i+1}: {img_name}")

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue

    print("-" * 30)
    print("Ingestion and Preprocessing Summary:")
    print(f"Original files considered: {len(image_files)}")
    print(f"Skipped files (mask missing or error): {skipped_count}")
    print(f"Total processed & saved tiles/images: {processed_count}")
    print(f"Ingestion complete. Processed data saved to {processed_dir}")
    print("-" * 30)


if __name__ == "__main__":
    run_ingestion_pipeline(
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        num_samples=NUM_SAMPLES,
        target_size=TARGET_SIZE,
        mask_threshold=MASK_THRESHOLD,
        num_augmentations=NUM_AUGMENTATIONS_PER_SAMPLE
    )
