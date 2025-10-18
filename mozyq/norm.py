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

    out_folder.mkdir(parents=True, exist_ok=True)

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
            with Image.open(image_path) as img:
                width, height = img.size

                if width < min_width or height < min_height:
                    skipped_count += 1
                    continue

                if img.mode != 'RGB':
                    img = img.convert('RGB')

                aspect_ratio = height / width
                new_height = int(target_width * aspect_ratio)
                img_resized = img.resize(
                    (target_width, new_height),
                    Image.Resampling.LANCZOS)

                if new_height >= target_height:
                    top = (new_height - target_height) // 2
                    bottom = top + target_height
                    img_cropped = img_resized.crop(
                        (0, top, target_width, bottom))
                else:
                    skipped_count += 1
                    continue

                output_path = out_folder / f'{i:04d}.jpg'
                img_cropped.save(output_path, 'JPEG', quality=95)
                processed_count += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            skipped_count += 1
            continue

    print(
        f"Processing complete: {processed_count} images processed, {skipped_count} images skipped")
