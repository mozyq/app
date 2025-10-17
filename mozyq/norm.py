from pathlib import Path

from PIL import Image
from tqdm import tqdm


def normalize(
    *,
    in_folder: Path,
    out_folder: Path,
    min_width: int,
    min_height: int,
    target_width: int,
        target_height: int):
    """
    Filter, resize, and crop images from input folder to output folder.

    Args:
        in_folder: Source directory containing images
        out_folder: Destination directory for processed images
        min_width: Minimum width threshold for filtering
        min_height: Minimum height threshold for filtering
        target_width: Target width after resizing
        target_height: Target height after cropping
    """
    # Create output folder if it doesn't exist
    out_folder.mkdir(parents=True, exist_ok=True)

    # Get all image files (common image extensions)
    image_extensions = {'.jpg', '.jpeg', '.png',
                        '.bmp', '.tiff', '.tif', '.webp'}
    image_files = [
        f for f in in_folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    processed_count = 0
    skipped_count = 0

    for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Open image and get dimensions
            with Image.open(image_path) as img:
                width, height = img.size

                # Skip images that don't meet minimum size requirements
                if width < min_width or height < min_height:
                    skipped_count += 1
                    continue

                # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize to target width while maintaining aspect ratio
                aspect_ratio = height / width
                new_height = int(target_width * aspect_ratio)
                img_resized = img.resize(
                    (target_width, new_height), Image.Resampling.LANCZOS)

                # Crop to target height (center crop)
                if new_height >= target_height:
                    # Crop from center
                    top = (new_height - target_height) // 2
                    bottom = top + target_height
                    img_cropped = img_resized.crop(
                        (0, top, target_width, bottom))
                else:
                    # If resized height is less than target, pad or skip
                    # For now, we'll skip such images
                    skipped_count += 1
                    continue

                # Save to output folder with same filename
                output_path = out_folder / f'{i:04d}.jpg'
                img_cropped.save(output_path, 'JPEG', quality=95)
                processed_count += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            skipped_count += 1
            continue

    print(
        f"Processing complete: {processed_count} images processed, {skipped_count} images skipped")
